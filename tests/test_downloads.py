import pytest
import requests_mock
from pathlib import Path
from aid2go.data.downloads import download_file_resumable

def test_download_file_fresh(tmp_path: Path):
    target = tmp_path / "sample.txt"
    test_url = "https://example.com/sample.txt"
    content = b"Hello, Protein Function Prediction!"

    with requests_mock.Mocker() as m:
        m.get(test_url, content=content, headers={"Content-Length": str(len(content))})
        res_path = download_file_resumable(test_url, target, force=True)

    assert res_path.exists()
    assert res_path.read_bytes() == content

def test_download_file_cached(tmp_path: Path):
    target = tmp_path / "cached.txt"
    target.write_bytes(b"Existing content")
    test_url = "https://example.com/cached.txt"

    # Should skip download without making any network request
    with requests_mock.Mocker() as m:
        res_path = download_file_resumable(test_url, target, force=False)

    assert res_path.exists()
    assert res_path.read_bytes() == b"Existing content"

def test_download_file_resumable_range_206(tmp_path: Path):
    target = tmp_path / "large_file.dat"
    part_path = tmp_path / "large_file.dat.part"
    test_url = "https://example.com/large_file.dat"

    # Simulate existing 5 bytes
    part_path.write_bytes(b"Part1")
    remaining_content = b"_Part2_Completed"
    full_content = b"Part1_Part2_Completed"

    with requests_mock.Mocker() as m:
        def match_range(request):
            return request.headers.get("Range") == "bytes=5-"

        m.get(
            test_url,
            content=remaining_content,
            status_code=206,
            headers={
                "Content-Length": str(len(remaining_content)),
                "Content-Range": f"bytes 5-{len(full_content)-1}/{len(full_content)}"
            },
            additional_matcher=match_range
        )
        res_path = download_file_resumable(test_url, target, force=False)

    assert res_path.exists()
    assert res_path.read_bytes() == full_content
    assert not part_path.exists()

def test_download_file_fallback_on_200(tmp_path: Path):
    target = tmp_path / "unsupported_range.dat"
    part_path = tmp_path / "unsupported_range.dat.part"
    test_url = "https://example.com/unsupported_range.dat"

    # Simulate stale part file
    part_path.write_bytes(b"CorruptedStart")
    full_content = b"FreshFullContentFromZero"

    with requests_mock.Mocker() as m:
        # Remote server ignores Range and sends 200 with full content
        m.get(
            test_url,
            content=full_content,
            status_code=200,
            headers={"Content-Length": str(len(full_content))}
        )
        res_path = download_file_resumable(test_url, target, force=False)

    assert res_path.exists()
    assert res_path.read_bytes() == full_content
    assert not part_path.exists()
