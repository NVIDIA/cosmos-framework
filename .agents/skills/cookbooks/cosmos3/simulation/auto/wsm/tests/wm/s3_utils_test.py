import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from cosmos_auto.wm.s3_utils import S3ClientAdapter, _parse_credentials, fetch_file_list_from_s3_to_json


def test_boto3_adapter_uses_keyword_upload_api():
    client = MagicMock()
    stream = BytesIO(b"video")

    S3ClientAdapter(client).upload_stream("bucket", "prefix/video.mp4", stream, 5)

    client.put_object.assert_called_once_with(
        Bucket="bucket",
        Key="prefix/video.mp4",
        Body=stream,
        ContentLength=5,
        ContentType="video/mp4",
    )


def test_boto3_adapter_lists_every_page():
    client = MagicMock()
    paginator = client.get_paginator.return_value
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "prefix/a"}]},
        {"Contents": [{"Key": "prefix/b"}]},
    ]

    assert S3ClientAdapter(client).list_keys("bucket", "prefix/") == ["prefix/a", "prefix/b"]
    paginator.paginate.assert_called_once_with(Bucket="bucket", Prefix="prefix/")


def test_boto3_adapter_only_treats_not_found_as_missing():
    client = MagicMock()
    adapter = S3ClientAdapter(client)
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
        "HeadObject",
    )

    assert not adapter.exists("bucket", "missing")

    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
        "HeadObject",
    )
    try:
        adapter.exists("bucket", "forbidden")
    except ClientError:
        pass
    else:
        raise AssertionError("AccessDenied must propagate")


def test_credential_values_are_not_logged(tmp_path, capsys):
    credential_path = tmp_path / "credentials.json"
    credential_path.write_text(
        json.dumps(
            {
                "endpoint_url": "https://s3.example.test",
                "aws_access_key_id": "sensitive-access-key",
                "aws_secret_access_key": "sensitive-secret-key",
                "region_name": "test-region",
            }
        )
    )

    _parse_credentials(credential_path)

    output = capsys.readouterr().out
    assert "sensitive-access-key" not in output
    assert "sensitive-secret-key" not in output


def test_legacy_cli_listing_uses_sdk_without_logging_credentials(tmp_path, capsys):
    output_path = tmp_path / "keys.json"
    with (
        patch("cosmos_auto.wm.s3_utils.list_s3_keys", return_value=["prefix/a"]),
        pytest.warns(UserWarning, match="uses the credential-safe SDK path"),
    ):
        fetch_file_list_from_s3_to_json("s3://bucket/prefix", output_path.as_posix(), "sensitive-profile", method="cli")

    assert json.loads(output_path.read_text()) == ["prefix/a"]
    assert "sensitive-profile" not in capsys.readouterr().out
