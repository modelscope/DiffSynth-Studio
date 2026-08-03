from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from . import config

METADATA_FILENAME = "metadata.jsonl"
INFO_FILENAME = "_dataset_info.json"

DATASET_KINDS = ["image", "edit", "video", "audio"]


@dataclass
class DatasetInfo:
    name: str
    path: str
    kind: str
    num_items: int


def _dataset_dir(name: str) -> Path:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"非法数据集名: {name!r}")
    return config.DATASETS_ROOT / name


def _read_info(dir_: Path) -> Dict[str, Any]:
    p = dir_ / INFO_FILENAME
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_info(dir_: Path, info: Dict[str, Any]) -> None:
    (dir_ / INFO_FILENAME).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_datasets() -> List[DatasetInfo]:
    config.ensure_dirs()
    result: List[DatasetInfo] = []
    for p in sorted(config.DATASETS_ROOT.iterdir()):
        if not p.is_dir():
            continue
        info = _read_info(p)
        items = read_metadata(p.name)
        result.append(
            DatasetInfo(
                name=p.name,
                path=str(p),
                kind=info.get("kind", "image"),
                num_items=len(items),
            )
        )
    return result


def create_dataset(name: str, kind: str = "image") -> DatasetInfo:
    if kind not in DATASET_KINDS:
        raise ValueError(f"kind 必须是 {DATASET_KINDS} 之一")
    d = _dataset_dir(name)
    if d.exists():
        raise FileExistsError(f"数据集已存在: {name}")
    d.mkdir(parents=True, exist_ok=False)
    (d / METADATA_FILENAME).write_text("", encoding="utf-8")
    _write_info(d, {"kind": kind})
    return DatasetInfo(name=name, path=str(d), kind=kind, num_items=0)


def delete_dataset(name: str) -> None:
    d = _dataset_dir(name)
    if not d.exists():
        return
    shutil.rmtree(d)


def dataset_path(name: str) -> Path:
    d = _dataset_dir(name)
    if not d.exists():
        raise FileNotFoundError(f"数据集不存在: {name}")
    return d


def image_path(name: str, media_path: str) -> Path:
    dataset_dir = dataset_path(name).resolve()
    relative = Path(str(media_path).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"非法图像路径: {media_path!r}")
    path = (dataset_dir / relative).resolve()
    try:
        path.relative_to(dataset_dir)
    except ValueError as exc:
        raise ValueError(f"图像路径超出数据集目录: {media_path!r}") from exc
    if not path.is_file() or path.suffix.lower() not in config.IMAGE_EXTS:
        raise FileNotFoundError(f"图像不存在或格式不受支持: {media_path}")
    return path


def metadata_path(name: str) -> Path:
    return dataset_path(name) / METADATA_FILENAME


def read_metadata(name: str) -> List[Dict[str, Any]]:
    p = _dataset_dir(name) / METADATA_FILENAME
    if not p.is_file():
        return []
    items: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        media_path = next(
            (item.get(field) for field in ("image", "video", "audio") if item.get(field)),
            None,
        )
        if media_path and _is_archive_junk(Path(str(media_path))):
            continue
        items.append(item)
    return items


def write_metadata(name: str, items: List[Dict[str, Any]]) -> None:
    p = _dataset_dir(name) / METADATA_FILENAME
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def upsert_item(name: str, media_path: str, prompt: str, **extras: Any) -> None:
    items = read_metadata(name)
    field = _media_field(media_path)
    for it in items:
        if it.get(field) == media_path:
            it["prompt"] = prompt
            for k, v in extras.items():
                it[k] = v
            write_metadata(name, items)
            return
    row = {field: media_path, "prompt": prompt}
    row.update(extras)
    items.append(row)
    write_metadata(name, items)


def remove_item(name: str, media_path: str) -> None:
    items = [
        it for it in read_metadata(name)
        if all(it.get(field) != media_path for field in ("image", "video", "audio"))
    ]
    write_metadata(name, items)


def delete_media(name: str, file_names: List[str]) -> List[str]:
    d = dataset_path(name).resolve()
    targets: List[tuple[str, Path]] = []
    for file_name in dict.fromkeys(file_names):
        relative = Path(str(file_name).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"非法媒体路径: {file_name!r}")
        target = (d / relative).resolve()
        try:
            target.relative_to(d)
        except ValueError as exc:
            raise ValueError(f"媒体路径超出数据集目录: {file_name!r}") from exc
        if target.exists() and (not target.is_file() or not _is_media(target)):
            raise ValueError(f"不是可删除的媒体文件: {file_name!r}")
        targets.append((relative.as_posix(), target))

    selected = {file_name for file_name, _ in targets}
    items = [
        item for item in read_metadata(name)
        if all(item.get(field) not in selected for field in ("image", "video", "audio"))
    ]
    deleted: List[str] = []
    for file_name, target in targets:
        if target.is_file():
            target.unlink()
            deleted.append(file_name)
        caption = target.with_suffix(".txt")
        if caption.is_file():
            caption.unlink()
        parent = target.parent
        while parent != d:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    write_metadata(name, items)
    return deleted


def get_extra_input_keys(name: str) -> List[str]:
    excluded = {"image", "video", "audio", "prompt"}
    keys: set = set()
    for it in read_metadata(name):
        keys.update(it.keys())
    return sorted(k for k in keys if k not in excluded)


def _media_field(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in config.VIDEO_EXTS:
        return "video"
    if suffix in config.AUDIO_EXTS:
        return "audio"
    return "image"


def _safe_relative_path(name: str) -> Path:
    raw = Path(name.replace("\\", "/"))
    if raw.is_absolute() or ".." in raw.parts or not raw.parts:
        raise ValueError(f"压缩包包含非法路径: {name!r}")
    return Path(*(_safe_name(part) for part in raw.parts))


def _is_archive_junk(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        "__macosx" in lowered_parts
        or name.startswith("._")
        or name in {".ds_store", "thumbs.db", "desktop.ini"}
    )


def _strip_common_archive_root(paths: List[Path]) -> List[Path]:
    if not paths or any(len(path.parts) < 2 for path in paths):
        return paths
    roots = {path.parts[0] for path in paths}
    if len(roots) != 1:
        return paths
    return [Path(*path.parts[1:]) for path in paths]


def _copy_archive_member(stream, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        shutil.copyfileobj(stream, out)


def _extract_zip(src: Path, target_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    with zipfile.ZipFile(src) as archive:
        entries = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"压缩包包含不支持的链接文件: {info.filename!r}")
            relative = _safe_relative_path(info.filename)
            if _is_archive_junk(relative) or relative.name == METADATA_FILENAME:
                continue
            entries.append((info, relative))
        normalized = _strip_common_archive_root([relative for _, relative in entries])
        for (info, _), relative in zip(entries, normalized):
            target = target_dir / relative
            with archive.open(info) as member:
                _copy_archive_member(member, target)
            extracted.append(relative)
    return extracted


def _extract_tar(src: Path, target_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    with tarfile.open(src) as archive:
        entries = []
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"压缩包包含不支持的特殊文件: {member.name!r}")
            relative = _safe_relative_path(member.name)
            if _is_archive_junk(relative) or relative.name == METADATA_FILENAME:
                continue
            entries.append((member, relative))
        normalized = _strip_common_archive_root([relative for _, relative in entries])
        for (member, _), relative in zip(entries, normalized):
            source = archive.extractfile(member)
            if source is None:
                continue
            with source:
                _copy_archive_member(source, target_dir / relative)
            extracted.append(relative)
    return extracted


def _sync_media_metadata(name: str, media_paths: List[Path]) -> None:
    if not media_paths:
        return
    d = dataset_path(name)
    items = read_metadata(name)
    index = {
        (field, str(item.get(field))): item
        for item in items
        for field in ("image", "video", "audio")
        if item.get(field)
    }
    changed = False
    for relative in media_paths:
        rel_name = relative.as_posix()
        field = _media_field(rel_name)
        caption_path = (d / relative).with_suffix(".txt")
        prompt = ""
        if caption_path.is_file():
            prompt = caption_path.read_text(encoding="utf-8", errors="replace").strip()
        item = index.get((field, rel_name))
        if item is None:
            item = {field: rel_name, "prompt": prompt}
            items.append(item)
            index[(field, rel_name)] = item
            changed = True
        elif caption_path.is_file() and item.get("prompt") != prompt:
            item["prompt"] = prompt
            changed = True
    if changed:
        write_metadata(name, items)

def add_files(name: str, files: List[Path]) -> List[str]:
    d = dataset_path(name)
    saved: List[str] = []
    media_paths: List[Path] = []
    for src in files:
        src = Path(src)
        suffix = "".join(src.suffixes).lower()
        if suffix in {".zip"}:
            extracted = _extract_zip(src, d)
            saved.append(f"[解压] {src.name}")
            media_paths.extend(p for p in extracted if _is_media(p))
            continue
        if suffix in {".tar", ".tar.gz", ".tgz"} or src.name.endswith(".tar.gz"):
            extracted = _extract_tar(src, d)
            saved.append(f"[解压] {src.name}")
            media_paths.extend(p for p in extracted if _is_media(p))
            continue
        target = d / _safe_name(src.name)
        shutil.copyfile(src, target)
        saved.append(target.name)
        if _is_media(target):
            media_paths.append(Path(target.name))
    _sync_media_metadata(name, media_paths)
    return saved


def _safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in name)


def _is_media(p: Path) -> bool:
    s = p.suffix.lower()
    return s in config.IMAGE_EXTS or s in config.VIDEO_EXTS or s in config.AUDIO_EXTS


def list_media(name: str) -> List[str]:
    d = dataset_path(name)
    files = []
    for p in sorted(d.rglob("*")):
        relative = p.relative_to(d)
        if p.is_file() and _is_media(p) and not _is_archive_junk(relative):
            files.append(relative.as_posix())
    return files
