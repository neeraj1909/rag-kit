"""Deterministic local-filesystem source acquisition."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlparse

from ragkit.domain import (
    AssetRef,
    IntegrityError,
    LimitExceededError,
    ProviderError,
    UnsupportedCapabilityError,
)
from ragkit.ports import AcquiredAsset, SourceConnector, SourceRequest


class FilesystemSourceConnector(SourceConnector):
    """Read one file or a recursively enumerated directory in stable path order."""

    def fetch(self, request: SourceRequest) -> tuple[AcquiredAsset, ...]:
        path = _local_path(request.source_uri)
        paths: tuple[Path, ...]
        try:
            if path.is_symlink():
                raise IntegrityError(f"symbolic link sources are not acquired: {path}")
            if path.is_file():
                paths = (path,)
            elif path.is_dir():
                entries = tuple(sorted(path.rglob("*")))
                linked = next((item for item in entries if item.is_symlink()), None)
                if linked is not None:
                    raise IntegrityError(
                        f"symbolic links inside source directories are not acquired: {linked}"
                    )
                paths = tuple(
                    item
                    for item in entries
                    if item.is_file() and "__pycache__" not in item.relative_to(path).parts
                )
            else:
                raise ProviderError(f"source does not exist: {path}")
        except OSError as error:
            raise ProviderError(f"cannot inspect source: {path}", cause=error) from error

        if len(paths) > request.max_assets:
            raise LimitExceededError(f"asset count {len(paths)} exceeds limit {request.max_assets}")

        assets: list[AcquiredAsset] = []
        for item in paths:
            try:
                size = item.stat().st_size
                if size > request.max_bytes_per_asset:
                    raise LimitExceededError(
                        f"asset size {size} exceeds limit {request.max_bytes_per_asset}: {item}"
                    )
                content = item.read_bytes()
            except LimitExceededError:
                raise
            except OSError as error:
                raise ProviderError(f"cannot read source asset: {item}", cause=error) from error
            if len(content) != size:
                raise IntegrityError(f"asset changed during acquisition: {item}")
            if not content:
                raise IntegrityError(f"empty assets are not accepted: {item}")
            uri = item.resolve().as_uri()
            digest = sha256(content).hexdigest()
            reference = AssetRef(
                asset_id=f"asset-{sha256(uri.encode()).hexdigest()}",
                media_type=_media_type(item),
                sha256=digest,
                uri=uri,
                size_bytes=size,
            )
            assets.append(AcquiredAsset(reference, content))
        return tuple(assets)


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise UnsupportedCapabilityError(
            f"unsupported source scheme: {parsed.scheme}", capability="source_scheme"
        )
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        raise UnsupportedCapabilityError(
            "remote file authorities are unsupported", capability="file_authority"
        )
    raw_path = unquote(parsed.path) if parsed.scheme else uri
    return Path(raw_path).expanduser().absolute()


def _media_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    return {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".eml": "message/rfc822",
        ".py": "text/x-python",
        ".js": "text/javascript",
        ".ts": "text/typescript",
        ".java": "text/x-java-source",
        ".go": "text/x-go",
        ".rs": "text/x-rust",
        ".c": "text/x-c",
        ".h": "text/x-c",
        ".cpp": "text/x-c++",
        ".json": "application/json",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".toml": "application/toml",
    }.get(suffix, "application/octet-stream")
