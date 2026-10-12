import hashlib

import httpx
import pytest
import respx

from pynatlas import models

CONTENT = b"GGUF" + bytes(range(256)) * 40
FILENAME = models.gguf_filename("Q4_K_M")
TREE = f"https://huggingface.co/api/models/{models.GGUF_REPO}/tree/{models.GGUF_REVISION}"
FILE = f"https://huggingface.co/{models.GGUF_REPO}/resolve/{models.GGUF_REVISION}/{FILENAME}"


def mock_tree(sha=None):
    respx.get(TREE).mock(return_value=httpx.Response(200, json=[
        {"path": FILENAME, "lfs": {"oid": sha or hashlib.sha256(CONTENT).hexdigest(),
                                   "size": len(CONTENT)}},
    ]))


def serve_ranges(request):
    start = 0
    if "range" in request.headers:
        start = int(request.headers["range"].split("=")[1].rstrip("-"))
        return httpx.Response(206, content=CONTENT[start:])
    return httpx.Response(200, content=CONTENT)


@respx.mock
def test_fresh_download_is_verified(tmp_path):
    mock_tree()
    respx.get(FILE).mock(side_effect=serve_ranges)
    path = models.download(dest=tmp_path / FILENAME)
    assert path.read_bytes() == CONTENT
    assert not path.with_suffix(".part").exists()


@respx.mock
def test_resume_requests_only_the_missing_bytes(tmp_path):
    mock_tree()
    route = respx.get(FILE).mock(side_effect=serve_ranges)
    dest = tmp_path / FILENAME
    dest.with_suffix(".part").write_bytes(CONTENT[:1000])
    models.download(dest=dest)
    assert route.calls[0].request.headers["range"] == "bytes=1000-"
    assert dest.read_bytes() == CONTENT


@respx.mock
def test_server_ignoring_range_restarts_cleanly(tmp_path):
    mock_tree()
    respx.get(FILE).mock(return_value=httpx.Response(200, content=CONTENT))
    dest = tmp_path / FILENAME
    dest.with_suffix(".part").write_bytes(CONTENT[:1000])
    models.download(dest=dest)
    assert dest.read_bytes() == CONTENT


@respx.mock
def test_complete_part_file_is_not_downloaded_again(tmp_path):
    mock_tree()
    route = respx.get(FILE).mock(side_effect=serve_ranges)
    dest = tmp_path / FILENAME
    dest.with_suffix(".part").write_bytes(CONTENT)
    models.download(dest=dest)
    assert not route.called
    assert dest.read_bytes() == CONTENT


@respx.mock
def test_corrupt_download_is_rejected_and_removed(tmp_path):
    mock_tree(sha="0" * 64)
    respx.get(FILE).mock(side_effect=serve_ranges)
    dest = tmp_path / FILENAME
    with pytest.raises(models.DownloadError, match="failed verification"):
        models.download(dest=dest)
    assert not dest.exists()
    assert not dest.with_suffix(".part").exists()


def test_existing_model_is_reused_without_network(tmp_path):
    dest = tmp_path / FILENAME
    dest.write_bytes(CONTENT)
    with respx.mock(assert_all_called=False) as mock:
        assert models.download(dest=dest) == dest
        assert not mock.calls


def test_unknown_quant_rejected():
    with pytest.raises(ValueError, match="Choose from"):
        models.gguf_filename("Q1")


@respx.mock
def test_progress_callback_reports_bytes(tmp_path):
    mock_tree()
    respx.get(FILE).mock(side_effect=serve_ranges)
    seen = []
    models.download(dest=tmp_path / FILENAME, progress=lambda done, total: seen.append(done))
    assert seen[-1] == len(CONTENT)


def test_missing_extra_names_the_install_command():
    from pynatlas._extras import MissingExtraError, require

    with pytest.raises(MissingExtraError, match=r"pip install 'pynatlas\[server\]'"):
        require("a_module_that_does_not_exist", "server")
