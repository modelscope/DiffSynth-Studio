#!/usr/bin/env python3
"""Check DiffSynth metadata, local paths and basic media readability; never modify data.

The actual model_training/train.py operators remain the final authority.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

# Hints for reporting failed probes, not a list of formats the framework supports.
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff', '.jfif', '.avif'}
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm'}
AUDIO_SUFFIXES = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.aiff'}
OPERATOR_FILE_SUFFIXES = {'.pth', '.pt', '.safetensors', '.npy', '.npz'}
REFERENCE_FIELDS = {'image': ('image',), 'video': ('video',), 'audio': ('audio',),
                    'video_audio': ('video', 'audio')}


class OperatorCheckRequired(Exception):
    """The generic probe cannot establish readability for this operator input."""


def parse_keys(text):
    return [x.strip() for x in text.split(',') if x.strip()]


def json_load(text, location):
    def invalid_constant(token):
        raise ValueError(f'Non-standard JSON constant {token} in {location}')
    try:
        return json.loads(text, parse_constant=invalid_constant)
    except json.JSONDecodeError as err:
        raise ValueError(f'Invalid JSON at {location}: {err.msg}') from err


def csv_row_lines(path):
    """Check column counts and track physical lines without changing caption contents."""
    row_lines, columns = [], None
    previous_limit = csv.field_size_limit()
    try:
        # Remove the stdlib's small field limit; pandas accepts long caption fields.
        field_limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(field_limit)
                break
            except OverflowError:
                field_limit //= 10
        with path.open(encoding='utf-8-sig', newline='') as stream:
            record_lines = []

            def tracked_lines():
                for line in stream:
                    record_lines.append(line)
                    yield line

            reader = csv.reader(tracked_lines())
            for values in reader:
                # pandas skips unquoted space/tab-only lines, including before the header.
                # Inspect raw lines so quoted whitespace and multiline captions stay intact.
                blank = all(not line.strip(' \t\r\n') for line in record_lines)
                record_lines.clear()
                if blank:
                    continue
                if columns is None:
                    columns = values
                    if not columns or len(columns) != len(set(columns)) or any(not c for c in columns):
                        raise ValueError('Missing, empty or duplicate CSV headers')
                else:
                    if len(values) != len(columns):
                        raise ValueError(f'CSV row {reader.line_num} has wrong number of columns')
                    row_lines.append(reader.line_num)
        if columns is None:
            raise ValueError('Missing CSV headers')
    finally:
        csv.field_size_limit(previous_limit)
    return row_lines


def iter_records(path):
    suffix = path.suffix
    if suffix == '.json':
        content = json_load(path.read_text(encoding='utf-8'), str(path))
        if not isinstance(content, list):
            raise ValueError('JSON metadata must be a top-level array of records')
        for index, record in enumerate(content, 1):
            yield index, record
    elif suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    raise ValueError(f'Blank JSONL line {line_no} (UnifiedDataset does not skip it)')
                yield line_no, json_load(line, f'line {line_no}')
    elif suffix.lower() == '.csv':
        # Check structure before pandas can rename duplicate headers or infer an index.
        row_lines = csv_row_lines(path)
        try:
            import pandas
        except ImportError as err:
            raise RuntimeError('CSV checks require pandas from the target training environment') from err
        # Match UnifiedDataset, including NA handling and whole-column type inference.
        frame = pandas.read_csv(path)
        if len(frame) != len(row_lines):
            raise ValueError('CSV record count differs between structural and pandas parsing')
        for index, line in enumerate(row_lines):
            yield line, frame.iloc[index].to_dict()
    else:
        raise ValueError('This checker supports .csv (case-insensitive), lowercase .json and .jsonl; UnifiedDataset matches JSON extensions case-sensitively')


def is_missing(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


def is_null(value):
    return value is None or isinstance(value, float) and math.isnan(value)


def file_references(value, field):
    """Recognize simple paths, ordered image lists, typed H3 references and track maps."""
    if isinstance(value, str) and value.strip():
        return [(value, None)]
    if isinstance(value, list) and value:
        if any(isinstance(item, list) for item in value):
            raise ValueError(f'{field}: nested path lists need a model-specific operator check')
        refs = []
        for item in value:
            refs.extend(file_references(item, field))
        return refs
    if isinstance(value, dict) and value:
        if 'type' in value:
            kind = value['type']
            if not isinstance(kind, str) or kind not in REFERENCE_FIELDS:
                raise ValueError(f'{field}: unsupported typed reference {kind!r}')
            refs = []
            for key in REFERENCE_FIELDS[kind]:
                if not isinstance(value.get(key), str) or not value[key].strip():
                    raise ValueError(f'{field}: {kind} reference requires a {key} path')
                refs.append((value[key], key))
            return refs
        # Track maps are valid static path collections, but their actual operators
        # may have stricter requirements; only loader sampling can establish that.
        if all(isinstance(v, str) and v.strip() or v is None for v in value.values()):
            refs = [(v, None) for v in value.values() if v is not None]
            if refs:
                return refs
    raise ValueError(f'{field}: expected path, nonempty path list, typed reference or path map; check the target operator')


def expected_kind(field):
    if field in ('video', 'input_video', 'animate_pose_video', 'animate_face_video', 'in_context_videos') or field.endswith('_video'):
        return 'video'
    if field in ('audio', 'input_audio', 'reference_audio') or field.endswith('_audio'):
        return 'audio'
    if field in ('image', 'edit_image', 'context_image', 'layer_input_image') or field.endswith('_image'):
        return 'image'
    return None


def media_check(path, media_hint=None):
    suffix = path.suffix.lower()
    if suffix in OPERATOR_FILE_SUFFIXES:
        raise OperatorCheckRequired('Serialized tensor/latent/cache: validate with the trusted target operator; no deserialization was attempted')
    try:
        from PIL import Image, ImageSequence, UnidentifiedImageError
    except ImportError:
        if media_hint == 'image' or suffix in IMAGE_SUFFIXES:
            raise
    else:
        # Like LoadImage, identify images by content, including unfamiliar suffixes.
        try:
            image = Image.open(path)
        except UnidentifiedImageError:
            if suffix in IMAGE_SUFFIXES:
                raise
            if media_hint == 'image':
                if suffix in VIDEO_SUFFIXES | AUDIO_SUFFIXES:
                    raise ValueError('Image field points to audio/video media')
                raise OperatorCheckRequired('Unrecognized image input; validate its type and readability with the target operator')
        else:
            with image:
                for frame in ImageSequence.Iterator(image):
                    frame.load()
            if media_hint == 'audio':
                raise ValueError('Audio condition points to a still image')
            return 'image_decoded'
    try:
        process = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'stream=codec_type,width,height,sample_rate,channels:format=duration',
             '-of', 'json', str(path)], capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError as err:
        raise RuntimeError('ffprobe not installed') from err
    if process.returncode:
        message = process.stderr[-1000:] or 'ffprobe failed'
        if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES | AUDIO_SUFFIXES:
            raise OperatorCheckRequired(f'Unrecognized media format; validate with the target operator: {message}')
        raise ValueError(message)
    streams = json_load(process.stdout, str(path)).get('streams', [])
    kinds = {s.get('codec_type') for s in streams}
    if not kinds or (media_hint in ('video', 'audio') and media_hint not in kinds):
        raise ValueError(f'Expected {media_hint or "a media"} stream, found {sorted(str(k) for k in kinds)}')
    if media_hint == 'image':
        raise ValueError('Image field points to audio/video media')
    return 'stream_metadata_probed_not_fully_decoded'


def validate(metadata, base, required, file_keys, text_keys, decode=False,
             id_key=None, allow_urls=False, max_errors=50, nullable_file_keys=(),
             allow_empty_file_keys=()):
    errors, rows, files_checked, urls_unverified = [], 0, 0, 0
    error_count, paths_seen, ids_seen, reused_count = 0, set(), set(), 0
    nullable_skipped, empty_lists_skipped = 0, 0
    pending_checks, pending_count = [], 0
    nullable_file_keys = set(nullable_file_keys)
    allow_empty_file_keys = set(allow_empty_file_keys)
    if max_errors < 1:
        raise ValueError('max_errors must be positive')
    if nullable_file_keys - set(file_keys):
        raise ValueError('Nullable file fields must also be listed in --file-keys')
    if nullable_file_keys & set(required):
        raise ValueError('--require means nonempty; nullable file fields cannot also be required')
    if allow_empty_file_keys - set(file_keys):
        raise ValueError('Empty-list file fields must also be listed in --file-keys')
    if allow_empty_file_keys & set(required):
        raise ValueError('--require means nonempty; empty-list file fields cannot also be required')

    def issue(line, field, message):
        nonlocal error_count
        error_count += 1
        if len(errors) < max_errors:
            errors.append({'row': line, 'field': field, 'error': message})

    def defer(line, field, path, reason):
        nonlocal pending_count
        pending_count += 1
        if len(pending_checks) < max_errors:
            pending_checks.append({'row': line, 'field': field, 'path': str(path), 'reason': reason})

    if not base.is_dir():
        raise ValueError(f'Dataset base path is not a directory: {base}')
    for line, record in iter_records(metadata):
        rows += 1
        if not isinstance(record, dict):
            issue(line, None, 'Row must be a JSON object / CSV record')
            continue
        for field in required:
            if is_missing(record.get(field)):
                issue(line, field, 'Required field missing or empty')
        for field in text_keys:
            if field in record and not isinstance(record[field], str):
                issue(line, field, 'Text field must be a string')
        if id_key:
            value = record.get(id_key)
            if isinstance(value, (dict, list)) or is_missing(value):
                issue(line, id_key, 'Sample ID missing or invalid')
            elif str(value) in ids_seen:
                issue(line, id_key, 'Duplicate sample ID')
            else:
                ids_seen.add(str(value))
        for field in file_keys:
            if field not in record:
                continue
            if field in nullable_file_keys and is_null(record[field]):
                nullable_skipped += 1
                continue
            if field in allow_empty_file_keys and isinstance(record[field], list) and not record[field]:
                empty_lists_skipped += 1
                continue
            try:
                refs = file_references(record[field], field)
            except ValueError as err:
                issue(line, field, str(err))
                continue
            for raw, hint in refs:
                parsed = urlparse(raw)
                if parsed.scheme in ('http', 'https'):
                    if allow_urls:
                        urls_unverified += 1
                        defer(line, field, raw, 'Remote URL not fetched; validate with the actual loader')
                    else:
                        issue(line, field, 'URL not checked locally; use --allow-urls and verify with the actual loader')
                    continue
                path = Path(raw)
                path = (path if path.is_absolute() else base / path).resolve()
                if path in paths_seen:
                    reused_count += 1
                else:
                    paths_seen.add(path)
                if not path.is_file() or path.stat().st_size == 0:
                    issue(line, field, f'Missing or empty local file: {path}')
                    continue
                files_checked += 1
                if decode:
                    try:
                        media_check(path, hint or expected_kind(field))
                    except OperatorCheckRequired as err:
                        defer(line, field, path, str(err))
                    except (OSError, ImportError, RuntimeError, ValueError, subprocess.TimeoutExpired) as err:
                        issue(line, field, f'Media check failed for {path}: {err}')
    if rows == 0:
        issue(None, None, 'Empty dataset; UnifiedDataset cannot sample it')
    status = 'failed' if error_count else 'needs_loader_check' if pending_count else 'static_checked'
    return {'schema_version': 1, 'status': status, 'rows': rows, 'files_checked': files_checked,
            'reused_path_occurrences': reused_count, 'unchecked_urls': urls_unverified,
            'nullable_files_skipped': nullable_skipped,
            'empty_file_lists_skipped': empty_lists_skipped,
            'error_count': error_count, 'errors': errors, 'errors_truncated': error_count - len(errors),
            'pending_check_count': pending_count, 'pending_checks': pending_checks,
            'pending_checks_truncated': pending_count - len(pending_checks),
            'metadata_parser': 'pandas.read_csv' if metadata.suffix.lower() == '.csv' else 'json',
            'media_check': 'image_full_decode_or_ffprobe' if decode else 'existence_only',
            'loader_check': 'not_run',
            'check_scope': 'Local paths and image full decoding / ffprobe stream metadata only when --decode; '
                           'unknown/serialized inputs are deferred without deserialization; '
                           'operator output types, video frames, audio alignment and training consumption require real UnifiedDataset sampling.'}


def failure_report(error):
    return {'schema_version': 1, 'status': 'failed', 'rows': None, 'error_count': 1,
            'errors': [{'row': None, 'field': None, 'error': str(error)}],
            'errors_truncated': 0, 'pending_check_count': 0, 'pending_checks': [],
            'pending_checks_truncated': 0, 'loader_check': 'not_run'}


def render_report(report):
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--require', default='', help='Comma-separated required metadata fields')
    parser.add_argument('--file-keys', required=True, help='Comma-separated path-bearing fields; may include typed references')
    parser.add_argument('--text-keys', default='prompt,lyrics', help='String-only fields for this recipe; use an empty value if none, and validate structured prompts with their processor')
    parser.add_argument('--nullable-file-keys', default='', help='Optional file fields allowed to be null/CSV NaN by the target operator')
    parser.add_argument('--allow-empty-file-keys', default='', help='Optional file fields whose target operator accepts an empty top-level list')
    parser.add_argument('--id-key', help='Metadata field used as a stable unique sample ID')
    parser.add_argument('--decode', action='store_true', help='Full image decode; ffprobe for audio/video streams')
    parser.add_argument('--allow-urls', action='store_true', help='Report remote URLs as unverified, never fetch them')
    parser.add_argument('--max-errors', type=int, default=50, help='Maximum stored details for errors and pending checks; totals remain complete')
    parser.add_argument('--output', type=Path, help='New report file; existing files and links are never overwritten')
    args = parser.parse_args()
    if args.max_errors < 1:
        parser.error('--max-errors must be positive')
    # Refuse existing outputs before reading/decoding data, including dangling links.
    # Exclusive creation below also protects against another process creating it later.
    if args.output:
        try:
            if args.output.resolve() == args.metadata.resolve():
                raise ValueError('Report path must differ from the input metadata path')
            if args.output.exists() or args.output.is_symlink():
                raise ValueError('Report path already exists; choose a new path or omit --output')
        except (OSError, ValueError, RuntimeError) as err:
            print(render_report(failure_report(err)), end='')
            return 1
    try:
        report = validate(args.metadata, args.base.resolve(), parse_keys(args.require),
                          parse_keys(args.file_keys), parse_keys(args.text_keys), args.decode,
                          args.id_key, args.allow_urls, args.max_errors,
                          parse_keys(args.nullable_file_keys), parse_keys(args.allow_empty_file_keys))
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, csv.Error) as err:
        report = failure_report(err)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(render_report(report))
        except (OSError, ValueError) as err:
            print(render_report(failure_report(f'Cannot write report: {err}')), end='')
            return 1
    else:
        print(render_report(report), end='')
    return 0 if report['status'] == 'static_checked' else 1


if __name__ == '__main__':
    raise SystemExit(main())
