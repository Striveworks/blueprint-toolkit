"""Module for S3 directory synchronization and path mapping.

This module provides utilities and functions for managing interactions with S3,
with a focus on replicating a local directory structure to an S3 key prefix and
vice-versa.
"""

import os
from pathlib import Path

from mypy_boto3_s3.client import S3Client


def _file_path_to_s3_key(
    file_path: str,
    file_path_prefix_to_strip: str | Path,
    s3_key_prefix: str,
) -> str:
    file_suffix = os.path.relpath(file_path, file_path_prefix_to_strip)
    key_suffix = file_suffix.replace(os.sep, "/")
    return f"{s3_key_prefix}{key_suffix}"


def _s3_key_to_file_path(
    key: str,
    file_path_prefix: str | Path,
    s3_key_prefix: str,
) -> str:
    assert s3_key_prefix.endswith("/")
    stripped_key = key.replace(s3_key_prefix, "")
    rel_file_path = stripped_key.replace("/", os.sep)
    return os.path.join(file_path_prefix, rel_file_path)


def upload_directory_to_s3(
    s3_client: S3Client,
    src_dir: str | Path,
    bucket: str,
    key_prefix: str,
):
    """Upload a directory to an S3 bucket.

    Parameters
    ----------
    s3_client : S3Client
        Boto S3 client
    src_dir : str or Path
        Directory to upload from
    bucket : str
        S3 bucket name
    key_prefix : str
        The key prefix to upload the objects to. Must end with a slash.

    Raises
    ------
    ValueError
        If ``key_prefix`` does not end with a slash

    """
    if not key_prefix.endswith("/"):
        raise ValueError("key_prefix must end with a forward slash")
    for directory, _, file_names in os.walk(src_dir):
        for file_name in file_names:
            file_path = os.path.join(directory, file_name)
            key = _file_path_to_s3_key(
                file_path=file_path,
                file_path_prefix_to_strip=src_dir,
                s3_key_prefix=key_prefix,
            )
            s3_client.upload_file(
                Filename=file_path,
                Bucket=bucket,
                Key=key,
            )


def download_directory_from_s3(
    s3_client: S3Client,
    bucket: str,
    key_prefix: str,
    dst_dir: str | Path,
):
    """Download a directory from an S3 bucket.

    Parameters
    ----------
    s3_client : S3Client
        Boto S3 client
    bucket : str
        S3 bucket name
    key_prefix : str
        The key prefix of the objects to download. Must end with a slash.
    dst_dir : str or Path
        Directory to download to

    Raises
    ------
    ValueError
        If ``key_prefix`` does not end with a slash

    """
    if not key_prefix.endswith("/"):
        raise ValueError("key_prefix must end with a forward slash")
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)
    for page in pages:
        for object in page.get("Contents", []):
            assert "Key" in object, "expected prefix content to contain a key"
            key = object["Key"]
            file_path = _s3_key_to_file_path(
                key=key,
                file_path_prefix=dst_dir,
                s3_key_prefix=key_prefix,
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            s3_client.download_file(Bucket=bucket, Key=key, Filename=file_path)
