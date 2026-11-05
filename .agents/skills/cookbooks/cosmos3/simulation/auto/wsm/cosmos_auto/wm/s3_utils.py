import json
import os
import warnings
from configparser import ConfigParser
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Tuple, Union

import boto3
import numpy as np
from botocore.exceptions import ClientError
from minio import Minio
from minio.error import S3Error


def _parse_credentials(
    cred_path_or_profile: Union[str, Path] | None = None,
) -> Tuple[str | None, str | None, str | None, str | None]:
    """
    Parse credentials from file, AWS profile, or environment variables.

    Args:
        cred_path_or_profile: Path to the credentials file OR AWS profile name.
                              If it's a file path, load credentials from the file.
                              If it's not a file, treat it as an AWS profile name and load from ~/.aws/credentials and ~/.aws/config.
                              If None, read from environment variables.

    Returns:
        Tuple of (endpoint, access_key, secret_key, region)
    """
    if cred_path_or_profile is not None:
        cred_path = Path(cred_path_or_profile)

        # Check if it's a file path
        if cred_path.exists() and cred_path.is_file():
            # Load from credentials file
            print("Loading credentials from the configured file")
            with open(cred_path) as credentials_file:
                credentials = json.load(credentials_file)
            endpoint = credentials["endpoint_url"]
            access_key = credentials["aws_access_key_id"]
            secret_key = credentials["aws_secret_access_key"]
            region = credentials.get("region_name", None)
        else:
            # Treat as AWS profile name
            profile = str(cred_path_or_profile)
            print("Loading credentials from the configured AWS profile")

            credentials_file = Path.home() / ".aws" / "credentials"
            config_file = Path.home() / ".aws" / "config"

            # Parse credentials file
            credentials_parser = ConfigParser()
            if credentials_file.exists():
                credentials_parser.read(credentials_file)
            else:
                raise FileNotFoundError(f"AWS credentials file not found: {credentials_file}")

            # Parse config file
            config_parser = ConfigParser()
            if config_file.exists():
                config_parser.read(config_file)

            # Get credentials from credentials file
            if profile not in credentials_parser:
                raise ValueError(f"Profile '{profile}' not found in {credentials_file}")

            access_key = credentials_parser[profile].get("aws_access_key_id")
            secret_key = credentials_parser[profile].get("aws_secret_access_key")

            if not access_key or not secret_key:
                raise ValueError(f"Missing aws_access_key_id or aws_secret_access_key in profile '{profile}'")

            # First, try to get region and endpoint_url from credentials file
            # (Some users put everything in credentials file)
            region = credentials_parser[profile].get("region")
            endpoint = credentials_parser[profile].get("endpoint_url")

            # If not found in credentials file, try config file
            if not region or not endpoint:
                config_section = f"profile {profile}" if profile != "default" else profile

                if config_section in config_parser:
                    if not region:
                        region = config_parser[config_section].get("region")
                    if not endpoint:
                        endpoint = config_parser[config_section].get("endpoint_url")

            if endpoint:
                print(f"Loaded the configured AWS profile for region={region}")
            else:
                print(f"Loaded the configured AWS profile for region={region}, using the default AWS S3 endpoint")
    else:
        # Load from environment variables
        print("Loading S3 credentials from the environment")
        endpoint = os.getenv("AWS_ENDPOINT_URL")
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION")

    # If no endpoint specified, use AWS S3 default endpoint
    # Both boto3 and minio clients are compatible with S3 API
    if not endpoint:
        endpoint = "https://s3.amazonaws.com"

    return endpoint, access_key, secret_key, region


def file_exists_in_s3(s3_path: str, client: Union[Minio, boto3.client]):
    bucket_name, object_name = s3_path.removeprefix("s3://").split("/", 1)
    return S3ClientAdapter(client).exists(bucket_name, object_name)


def get_client(
    cred_path_or_profile: Union[str, Path] | None = None,
    backend: str = "boto3",  # "minio" or "boto3"
):
    """
    Get a client for uploading files to S3 with Minio or boto3.
    Args:
        cred_path_or_profile: Path to the credentials file OR AWS profile name.
                              If it's a file path, load credentials from the file.
                              If it's not a file, treat it as an AWS profile name and load from ~/.aws/credentials and ~/.aws/config.
                              If None, read from environment variables.
        backend: The backend to use, "minio" or "boto3".
    Returns:
        client: Minio or boto3 client.
    """
    # Parse credentials using the shared function
    # If endpoint is None, it will be set to AWS S3 default endpoint
    endpoint, access_key, secret_key, region = _parse_credentials(cred_path_or_profile)

    if backend == "minio":
        # Minio client is compatible with S3 API
        # Remove protocol prefix for Minio client
        if endpoint.startswith("https://"):
            endpoint = endpoint[8:]
        elif endpoint.startswith("http://"):
            endpoint = endpoint[7:]

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, region=region)
    elif backend == "boto3":
        # boto3 client
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return client


@dataclass(frozen=True)
class S3ClientAdapter:
    """Backend-neutral S3 operations used by rendering and resume."""

    client: Any

    def upload_stream(
        self,
        bucket_name: str,
        object_name: str,
        stream: BinaryIO,
        length: int,
        content_type: str = "video/mp4",
    ) -> None:
        if isinstance(self.client, Minio):
            self.client.put_object(
                bucket_name,
                object_name,
                stream,
                length=length,
                content_type=content_type,
            )
        else:
            self.client.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=stream,
                ContentLength=length,
                ContentType=content_type,
            )

    def upload_file(self, local_path: str, bucket_name: str, object_name: str) -> None:
        if isinstance(self.client, Minio):
            self.client.fput_object(bucket_name, object_name, local_path)
        else:
            self.client.upload_file(local_path, bucket_name, object_name)

    def download_file(self, bucket_name: str, object_name: str, local_path: str) -> None:
        if isinstance(self.client, Minio):
            self.client.fget_object(bucket_name, object_name, local_path)
        else:
            self.client.download_file(bucket_name, object_name, local_path)

    def delete(self, bucket_name: str, object_name: str) -> None:
        if isinstance(self.client, Minio):
            self.client.remove_object(bucket_name, object_name)
        else:
            self.client.delete_object(Bucket=bucket_name, Key=object_name)

    def list_keys(self, bucket_name: str, prefix: str) -> list[str]:
        if isinstance(self.client, Minio):
            return [obj.object_name for obj in self.client.list_objects(bucket_name, prefix=prefix, recursive=True)]

        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    def exists(self, bucket_name: str, object_name: str) -> bool:
        try:
            if isinstance(self.client, Minio):
                self.client.stat_object(bucket_name, object_name)
            else:
                self.client.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = error.response.get("Error", {}).get("Code")
            if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        except S3Error as error:
            if error.code in {"NoSuchKey", "NoSuchObject", "NotFound"}:
                return False
            raise


def get_s3_adapter(
    cred_path_or_profile: Union[str, Path] | None = None,
    backend: str = "boto3",
) -> S3ClientAdapter:
    """Construct the adapter while keeping credentials inside the S3 boundary."""
    return S3ClientAdapter(get_client(cred_path_or_profile, backend=backend))


def list_s3_keys(s3_folder: str, cred_path_or_profile: Union[str, Path] | None) -> list[str]:
    """List all object keys below an S3 URI through the SDK."""
    bucket_name, prefix = s3_folder.removeprefix("s3://").split("/", 1)
    return get_s3_adapter(cred_path_or_profile).list_keys(bucket_name, prefix)


def client_upload(
    local_path: str,
    s3_path: str,
    client: Union[Minio, boto3.client],
):
    """
    Upload a file to S3.
    Args:
        local_path: Path to the local file.
        s3_path: Path to the S3 file.
        client: Minio or boto3 client.
    """
    bucket_name, object_name = s3_path.replace("s3://", "").split("/", 1)

    S3ClientAdapter(client).upload_file(local_path, bucket_name, object_name)


def client_download(
    s3_path: str,
    local_path: str,
    client: Union[Minio, boto3.client],
):
    """
    Download a file from S3.
    Args:
        s3_path: Path to the S3 file.
        local_path: Path to the local file.
        client: Minio or boto3 client.
    """
    bucket_name, object_name = s3_path.replace("s3://", "").split("/", 1)

    S3ClientAdapter(client).download_file(bucket_name, object_name, local_path)


def client_delete(
    s3_path: str,
    client: Union[Minio, boto3.client],
):
    """
    Delete a file from S3.
    Args:
        s3_path: Path to the S3 file.
        client: Minio or boto3 client.
    """
    bucket_name, object_name = s3_path.replace("s3://", "").split("/", 1)

    S3ClientAdapter(client).delete(bucket_name, object_name)


def upload_files_to_s3_storage(
    local_paths: list[str] | np.ndarray,
    s3_paths: list[str] | np.ndarray,
    cred_path_or_profile: Union[str, Path] | None = None,
    backend: str = "minio",
):
    """
    Upload a list of files to S3.
    Args:
        local_paths: List of paths to the local files.
        s3_paths: List of paths to the S3 files.
        cred_path_or_profile: Path to the credentials file OR AWS profile name.
        backend: The backend to use, "minio" or "boto3".
    """
    assert len(local_paths) == len(s3_paths), "The number of local paths and S3 paths must be the same"
    client = get_client(cred_path_or_profile, backend)

    if isinstance(local_paths, np.ndarray):
        local_paths = local_paths.tolist()
    if isinstance(s3_paths, np.ndarray):
        s3_paths = s3_paths.tolist()

    for local_path, s3_path in zip(local_paths, s3_paths):
        client_upload(local_path, s3_path, client)


def upload_folder_to_s3_storage_mpire(
    local_folder: str,
    s3_folder: str,
    num_workers: int = 32,
    cred_path_or_profile: Union[str, Path] | None = None,
    chunk_num: int = 1000,
    backend: str = "minio",
):
    """
    Upload a folder to S3 with multiprocessing.
    Args:
        local_path: Path to the local folder.
        s3_path: Path to the S3 folder.
        num_workers: Number of workers to use.
        cred_path_or_profile: Path to the credentials file OR AWS profile name.
        chunk_num: Number of chunks to split the files into.
        backend: The backend to use, "minio" or "boto3".
    """
    import mpire

    # remove trailing slashes
    local_folder = local_folder.rstrip("/")
    s3_folder = s3_folder.rstrip("/")

    # list all files in the local folder, including subfolders
    local_files = []
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_files.append(os.path.join(root, file))

    # print chunk size
    print(f"Total number of files: {len(local_files)}")
    print(f"Number of workers: {num_workers}")
    print(f"Average chunk size: {len(local_files) // chunk_num}")

    # manual devide the files into chunks. using numpy
    local_files = np.array(local_files)
    s3_files = [f.replace(local_folder, s3_folder) for f in local_files]
    local_files_chunks_array = np.array_split(local_files, chunk_num)
    s3_files_chunks_array = np.array_split(s3_files, chunk_num)

    with mpire.WorkerPool(num_workers) as pool:
        pool.map_unordered(
            partial(upload_files_to_s3_storage, cred_path_or_profile=cred_path_or_profile, backend=backend),
            zip(local_files_chunks_array, s3_files_chunks_array),
            iterable_len=len(local_files_chunks_array),
            progress_bar=True,
        )


def fetch_file_list_from_s3_to_json(
    s3_folder: str,
    output_json_path: str,
    cred_path_or_profile: Union[str, Path] | None,
    block: bool = True,
    method: str = "sdk",
):
    """Fetch an S3 prefix listing with the SDK and write its keys as JSON.

    ``method`` is retained for compatibility with existing callers. All supported
    values use the same paginated SDK implementation; the old shell-based path is
    intentionally disabled so credentials never enter logs or process arguments.
    """
    if method not in {"sdk", "paginator", "list_objects_v2", "cli"}:
        raise ValueError(f"Invalid method: {method}")
    if not block:
        warnings.warn("Non-blocking S3 listing is no longer supported; listing synchronously.", stacklevel=2)
    if method == "cli":
        warnings.warn("method='cli' is deprecated and now uses the credential-safe SDK path.", stacklevel=2)

    all_keys = list_s3_keys(s3_folder, cred_path_or_profile)
    with open(output_json_path, "w") as output_file:
        json.dump(all_keys, output_file, indent=4)
    print(f"Found {len(all_keys)} objects and saved the listing to {output_json_path}")
