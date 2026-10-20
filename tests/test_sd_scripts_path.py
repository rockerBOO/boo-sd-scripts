import sys

import pytest

from boo_sd_scripts import _sd_scripts


def test_default_path_is_repo_submodule(monkeypatch):
    monkeypatch.delenv("SD_SCRIPTS_PATH", raising=False)
    path = _sd_scripts.sd_scripts_path()
    assert path.name == "sd-scripts"
    assert (path / "train_network.py").is_file()


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SD_SCRIPTS_PATH", str(tmp_path))
    assert _sd_scripts.sd_scripts_path() == tmp_path.resolve()


def test_ensure_on_path_inserts_front(monkeypatch, tmp_path):
    (tmp_path / "train_network.py").write_text("")
    monkeypatch.setenv("SD_SCRIPTS_PATH", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    _sd_scripts.ensure_on_path()
    assert sys.path[0] == str(tmp_path.resolve())


def test_ensure_on_path_rejects_wrong_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SD_SCRIPTS_PATH", str(tmp_path))
    with pytest.raises(ImportError, match="train_network.py"):
        _sd_scripts.ensure_on_path()
