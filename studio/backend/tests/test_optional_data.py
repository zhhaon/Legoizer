"""Release contract: .data is optional and never disables automatic acquisition."""
from pathlib import Path
import warnings

import pytest

from studio.backend.executor import provision
from utils.config_init import DataInit


@pytest.fixture(autouse=True)
def isolated_data_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataInit, "_d", None)


def test_missing_data_is_silent_and_uses_defaults():
    with warnings.catch_warnings(record=True) as caught:
        assert DataInit.get("mind", default="") == ""
    assert not caught


def test_commented_example_and_equals_in_paths():
    Path(".data").write_text("# Optional local data\n\nmind = /datasets/version=1\n")
    assert DataInit.get("mind") == "/datasets/version=1"


def test_invalid_entry_has_an_actionable_line_number():
    Path(".data").write_text("# Optional\nbroken entry\n")
    with pytest.raises(ValueError, match="line 2"):
        DataInit.parse()


def test_valid_local_mind_is_reused(tmp_path, monkeypatch):
    for split in ("train", "dev"):
        (tmp_path / split).mkdir()
        for name in ("news.tsv", "behaviors.tsv"):
            (tmp_path / split / name).touch()
    Path(".data").write_text(f"mind = {tmp_path}\n")
    monkeypatch.setattr(provision, "download", lambda *_: pytest.fail("Local data must not download"))
    assert provision.raw_mind() == tmp_path


@pytest.mark.parametrize("invalid_override", [False, True])
def test_mind_auto_downloads_without_a_valid_override(tmp_path, monkeypatch, invalid_override):
    if invalid_override:
        Path(".data").write_text("mind = /does/not/exist\n")
    monkeypatch.setattr(provision, "DOWNLOADS", tmp_path / "downloads")
    calls = []

    def download(name, urls):
        calls.append(name)
        assert urls[0] == f"https://huggingface.co/datasets/Recommenders/MIND/resolve/main/{name}"
        return tmp_path / name

    monkeypatch.setattr(provision, "download", download)
    monkeypatch.setattr(provision, "extract_members", lambda *_: None)
    assert provision.raw_mind() == tmp_path / "downloads/mind"
    assert calls == ["MINDsmall_train.zip", "MINDsmall_dev.zip"]
