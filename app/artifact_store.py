# app/artifact_store.py
from __future__ import annotations

import logging
import os
import pathlib
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Protocol, Tuple
from uuid import UUID


logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    # Prevent path traversal + keep URLs clean
    name = name.strip().replace("\\", "/")
    name = name.split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name or "artifact"


def _join_key(*parts: str) -> str:
    return "/".join(p.strip("/").replace("\\", "/") for p in parts if p)


class ArtifactStore(Protocol):
    """
    Blob storage interface.
    - Returns a durable blob_path string you store in DB (Artifact.blob_path).
    - For local dev, blob_path can be a file path.
    - For GCS, blob_path is gs://bucket/key.

    The API layer can either:
    - return signed URLs (preferred for GCS), or
    - stream bytes via GET /artifacts/{id}/download by reading through this store.
    """

    def put_bytes(
        self,
        *,
        job_id: UUID,
        kind: str,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Tuple[str, int]:
        ...

    def put_file(
        self,
        *,
        job_id: UUID,
        kind: str,
        filename: str,
        local_path: str,
        content_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Tuple[str, int]:
        ...

    def get_bytes(self, blob_path: str) -> bytes:
        ...

    def signed_url(self, blob_path: str, *, expires_in: int = 3600) -> Optional[str]:
        ...

    def delete_blob(self, blob_path: str) -> None:
        ...


# -------------------------
# Local filesystem store (dev)
# -------------------------

@dataclass
class LocalArtifactStore:
    """
    Stores artifacts under a root directory like:
      {root}/jobs/{job_id}/{kind}/{label?}/{filename}

    blob_path returned is an absolute filesystem path (string).
    """
    root_dir: str

    def __post_init__(self) -> None:
        pathlib.Path(self.root_dir).mkdir(parents=True, exist_ok=True)

    def _target_path(self, *, job_id: UUID, kind: str, filename: str, label: Optional[str]) -> pathlib.Path:
        safe_filename = _safe_name(filename)
        safe_kind = _safe_name(kind)
        safe_label = _safe_name(label) if label else None

        parts = ["jobs", str(job_id), safe_kind]
        if safe_label:
            parts.append(safe_label)
        parts.append(safe_filename)

        return pathlib.Path(self.root_dir, *parts)

    def put_bytes(
        self,
        *,
        job_id: UUID,
        kind: str,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Tuple[str, int]:
        path = self._target_path(job_id=job_id, kind=kind, filename=filename, label=label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return (str(path.resolve()), len(data))

    def put_file(
        self,
        *,
        job_id: UUID,
        kind: str,
        filename: str,
        local_path: str,
        content_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Tuple[str, int]:
        src = pathlib.Path(local_path)
        data = src.read_bytes()
        return self.put_bytes(job_id=job_id, kind=kind, filename=filename, data=data, content_type=content_type, label=label)

    def get_bytes(self, blob_path: str) -> bytes:
        return pathlib.Path(blob_path).read_bytes()

    def signed_url(self, blob_path: str, *, expires_in: int = 3600) -> Optional[str]:
        # Local paths generally aren't publicly accessible; return None so API can stream
        return None

    def delete_blob(self, blob_path: str) -> None:
        try:
            p = pathlib.Path(blob_path)
            if p.exists():
                p.unlink()
        except Exception:
            # Best-effort deletion; caller can log if needed
            return


# -------------------------
# Google Cloud Storage store (prod)
# -------------------------

@dataclass
class GCSArtifactStore:
    """
    Stores artifacts in GCS:
      gs://{bucket}/{prefix}/jobs/{job_id}/{kind}/{label?}/{filename}

    Requires:
      pip install google-cloud-storage
    and Cloud Run service account permissions:
      storage.objects.create / storage.objects.get (and signBlob if using signed urls)
    """
    bucket: str
    prefix: str = ""  # optional, e.g. "prod" or "myapp"
    public: bool = False  # if you want to use public URLs instead of signed URLs

    def __post_init__(self) -> None:
        try:
            from google.cloud import storage  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "google-cloud-storage is required for GCSArtifactStore. "
                "Install with: pip install google-cloud-storage"
            ) from e
        self._storage = storage
        self._client = storage.Client()
        self._bucket = self._client.bucket(self.bucket)

    def _key(self, *, job_id: UUID, kind: str, filename: str, label: Optional[str]) -> str:
        safe_filename = _safe_name(filename)
        safe_kind = _safe_name(kind)
        safe_label = _safe_name(label) if label else None

        parts = []
        if self.prefix:
            parts.append(self.prefix.strip("/"))
        parts += ["jobs", str(job_id), safe_kind]
        if safe_label:
            parts.append(safe_label)
        parts.append(safe_filename)

        return _join_key(*parts)

    def put_bytes(
        self,
        *,
        job_id: UUID,
        kind: str,
        filename: str,
        data: bytes,
        content_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Tuple[str, int]:
        key = self._key(job_id=job_id, kind=kind, filename=filename, label=label)
        blob = self._bucket.blob(key)
        blob.upload_from_string(data, content_type=content_type or "application/octet-stream")

        if self.public:
            blob.make_public()

        blob_path = f"gs://{self.bucket}/{key}"
        return (blob_path, len(data))

    def put_file(
        self,
        *,
        job_id: UUID,
        kind: str,
        filename: str,
        local_path: str,
        content_type: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Tuple[str, int]:
        key = self._key(job_id=job_id, kind=kind, filename=filename, label=label)
        blob = self._bucket.blob(key)

        lp = pathlib.Path(local_path)
        size = lp.stat().st_size
        blob.upload_from_filename(str(lp), content_type=content_type or "application/octet-stream")

        if self.public:
            blob.make_public()

        blob_path = f"gs://{self.bucket}/{key}"
        return (blob_path, int(size))

    def get_bytes(self, blob_path: str) -> bytes:
        bucket, key = self._parse_gs_uri(blob_path)
        b = self._client.bucket(bucket).blob(key)
        return b.download_as_bytes()

    def signed_url(self, blob_path: str, *, expires_in: int = 3600) -> Optional[str]:
        """
        Returns a signed URL for GET access.
        If `public=True`, you can instead return the public URL.
        """
        bucket, key = self._parse_gs_uri(blob_path)
        blob = self._client.bucket(bucket).blob(key)

        if self.public:
            return blob.public_url

        # Cloud Run commonly uses token-only credentials, so fall back to proxy downloads
        # when object signing is unavailable.
        try:
            return blob.generate_signed_url(expiration=timedelta(seconds=expires_in), method="GET")
        except Exception as exc:
            logger.warning("Falling back to backend artifact download for %s: %s", blob_path, exc)
            return None

    def delete_blob(self, blob_path: str) -> None:
        bucket, key = self._parse_gs_uri(blob_path)
        b = self._client.bucket(bucket).blob(key)
        b.delete()

    @staticmethod
    def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
        # gs://bucket/key
        if not gs_uri.startswith("gs://"):
            raise ValueError(f"Not a gs:// uri: {gs_uri}")
        rest = gs_uri[len("gs://") :]
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise ValueError(f"Invalid gs:// uri: {gs_uri}")
        return bucket, key
