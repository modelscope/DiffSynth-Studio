#!/usr/bin/env python3
"""Decode inference outputs and check basic specs. Does not assess quality or AV synchronization."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess


FIELDS = ('width', 'height', 'frames', 'fps', 'sample_rate', 'channels', 'duration_seconds')
SPEC_KEYS = {'path', 'kind', 'id', 'require_audio', 'fps_tolerance', 'duration_tolerance_seconds', *FIELDS}


def run(command, timeout):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise ValueError(result.stderr[-2000:] or f'{command[0]} failed')
    return result.stdout


def decode(spec, path, timeout=120):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('Output missing or empty')
    kind = spec['kind']
    if kind == 'image':
        from PIL import Image
        with Image.open(path) as image:
            actual = dict(width=image.width, height=image.height, frames=getattr(image, 'n_frames', 1))
            for index in range(actual['frames']):
                image.seek(index)
                image.load()
            return actual
    if kind not in ('audio', 'video'):
        raise ValueError('kind must be image, audio or video')
    data = json.loads(run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(path)], timeout))
    video = next((s for s in data['streams'] if s['codec_type'] == 'video' and not s.get('disposition', {}).get('attached_pic')), None)
    audio = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
    primary = video if kind == 'video' else audio
    if primary is None or spec.get('require_audio') and audio is None:
        raise ValueError('Required media stream missing')
    command = ['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-map', f"0:{primary['index']}"]
    if kind == 'video' and spec.get('require_audio'):
        command += ['-map', f"0:{audio['index']}"]
    run(command + ['-f', 'null', '-'], timeout)
    actual = dict(has_video=video is not None, has_audio=audio is not None)
    duration = primary.get('duration')
    if duration in (None, 'N/A'):
        duration = data.get('format', {}).get('duration')
    if duration not in (None, 'N/A'):
        actual['duration_seconds'] = float(duration)
    if kind == 'video':
        actual.update(width=video['width'], height=video['height'])
        if video.get('nb_read_frames') not in (None, 'N/A'):
            actual['frames'] = int(video['nb_read_frames'])
        if video.get('avg_frame_rate') not in (None, '0/0', 'N/A'):
            actual['fps'] = float(Fraction(video['avg_frame_rate']))
        actual['fps_interpretation'] = 'average only; frame cadence not checked'
    if audio:
        actual.update(sample_rate=int(audio['sample_rate']), channels=int(audio['channels']))
    return actual


def positive_number(value, name, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    return value


def validate(manifest, base, timeout=120):
    if not isinstance(manifest, dict) or set(manifest) - {'outputs', 'count'}:
        raise ValueError('Manifest supports outputs and optional count only')
    outputs = manifest.get('outputs')
    if not isinstance(outputs, list) or not outputs:
        raise ValueError('outputs must be a nonempty list')
    if type(manifest.get('count', len(outputs))) is not int or manifest.get('count', len(outputs)) != len(outputs):
        raise ValueError('Output count does not match manifest')
    positive_number(timeout, 'timeout')
    results, seen = [], set()
    for spec in outputs:
        row = dict(path=spec.get('path') if isinstance(spec, dict) else None, status='failed', errors=[])
        try:
            if not isinstance(spec, dict) or set(spec) - SPEC_KEYS:
                raise ValueError('Unsupported output fields; helper checks basic media specs only')
            if not isinstance(spec.get('path'), str) or not spec['path'].strip():
                raise ValueError('Output path must be a nonempty string')
            if 'require_audio' in spec and (type(spec['require_audio']) is not bool or spec.get('kind') != 'video'):
                raise ValueError('require_audio must be a boolean for kind=video')
            path = (Path(base) / spec['path']).resolve()
            if path in seen:
                raise ValueError('Duplicate output path')
            seen.add(path)
            actual = row['actual'] = decode(spec, path, timeout)
            for field in FIELDS:
                if field not in spec:
                    continue
                expected = positive_number(spec[field], field)
                tolerance = (spec.get('fps_tolerance', 0.01) if field == 'fps' else
                             spec.get('duration_tolerance_seconds', 0.1) if field == 'duration_seconds' else 0)
                positive_number(tolerance, 'tolerance', allow_zero=True)
                observed = actual.get(field)
                if observed is None or not math.isfinite(observed) or abs(observed - expected) > tolerance:
                    row['errors'].append(f'{field}: expected {expected}, observed {observed}')
            row['status'] = 'failed' if row['errors'] else 'media_validated'
        except (FileNotFoundError, ImportError) as error:
            # A missing output is caught by decode as ValueError; this is a missing decoder.
            row.update(status='needs_dependency', errors=[str(error)])
        except subprocess.TimeoutExpired as error:
            row.update(status='verification_timeout', errors=[str(error)])
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            row['errors'].append(str(error))
        results.append(row)
    status = next((s for s in ('failed', 'verification_timeout', 'needs_dependency') if any(r['status'] == s for r in results)), 'media_validated')
    return dict(status=status, outputs=results, count=len(results), quality='not_reviewed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--timeout', type=float, default=120, help='Per probe/decode timeout in seconds')
    parser.add_argument('--output', type=Path, help='New report file; existing files and links are never overwritten')
    args = parser.parse_args()
    try:
        if args.output:
            if args.output.resolve() == args.manifest.resolve():
                raise ValueError('Report path must differ from the input manifest path')
            # Reject before decoding; this also protects media files and hard links.
            if args.output.exists() or args.output.is_symlink():
                raise ValueError('Report path already exists; choose a new path or omit --output')
        report = validate(json.loads(args.manifest.read_text()), args.manifest.resolve().parent, args.timeout)
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also rejects a file/link created after the check.
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(rendered)
        else:
            print(rendered, end='')
        return 0 if report['status'] == 'media_validated' else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
