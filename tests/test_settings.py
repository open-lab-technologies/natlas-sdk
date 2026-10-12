from pathlib import Path

import pytest

from pynatlas import NAtlas
from pynatlas.cli import main
from pynatlas.settings import ConfigError, Settings


def test_defaults_when_nothing_is_set():
    s = Settings.from_env({})
    assert (s.base_url, s.model, s.timeout, s.max_retries) == (
        "http://localhost:8000/v1", "n-atlas", 120.0, 2)
    assert s.api_key is None and s.model_path is None


def test_values_are_read_and_typed():
    s = Settings.from_env({
        "NATLAS_BASE_URL": "https://atlas.example.ng/v1/", "NATLAS_API_KEY": "k",
        "NATLAS_MODEL": "NCAIR1/N-ATLaS", "NATLAS_TIMEOUT": "30", "NATLAS_MAX_RETRIES": "0",
        "NATLAS_MODEL_PATH": "~/m.gguf", "NATLAS_CACHE": "/data/natlas",
    })
    assert s.base_url == "https://atlas.example.ng/v1"
    assert (s.timeout, s.max_retries) == (30.0, 0)
    assert s.model_path == Path("~/m.gguf").expanduser()
    assert s.cache_dir == Path("/data/natlas")


def test_empty_values_keep_defaults():
    blank = Settings.from_env({"NATLAS_BASE_URL": "  ", "NATLAS_TIMEOUT": ""})
    assert blank == Settings.from_env({})


@pytest.mark.parametrize(("name", "value", "message"), [
    ("NATLAS_BASE_URL", "localhost:8000", "must be an http"),
    ("NATLAS_BASE_URL", "ftp://host/v1", "must be an http"),
    ("NATLAS_TIMEOUT", "soon", "number of seconds"),
    ("NATLAS_TIMEOUT", "0", "greater than 0"),
    ("NATLAS_MAX_RETRIES", "2.5", "whole number"),
    ("NATLAS_MAX_RETRIES", "-1", "0 or more"),
])
def test_malformed_values_name_the_variable(name, value, message):
    with pytest.raises(ConfigError, match=message) as err:
        Settings.from_env({name: value})
    assert name in str(err.value)


def test_api_key_is_never_shown():
    s = Settings.from_env({"NATLAS_API_KEY": "super-secret"})
    assert "super-secret" not in repr(s)
    assert "super-secret" not in str(s.describe())


def test_client_validates_environment_at_construction(monkeypatch):
    monkeypatch.setenv("NATLAS_BASE_URL", "localhost:8000")
    with pytest.raises(ConfigError, match="NATLAS_BASE_URL"):
        NAtlas()


def test_explicit_arguments_override_environment(monkeypatch):
    monkeypatch.setenv("NATLAS_TIMEOUT", "5")
    monkeypatch.setenv("NATLAS_MAX_RETRIES", "4")
    c = NAtlas(base_url="http://other:9000/v1", timeout=60, max_retries=0)
    assert (c.base_url, c.timeout, c.max_retries) == ("http://other:9000/v1", 60, 0)
    assert NAtlas().timeout == 5.0
    with pytest.raises(ConfigError):
        NAtlas(base_url="not a url")


def test_config_command_masks_the_key(monkeypatch, capsys):
    monkeypatch.setenv("NATLAS_API_KEY", "super-secret")
    main(["config"])
    out = capsys.readouterr().out
    assert "NATLAS_API_KEY" in out and "super-secret" not in out
