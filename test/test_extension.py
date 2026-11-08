import inspect

import pytest
import torchvision._meta_registrations as meta_registrations
from torchvision import extension


def _raise_missing_module(lib_name):
    raise ImportError(f"Could not find module '{lib_name}'")


def test_load_library_caches_error(monkeypatch):
    monkeypatch.setattr(extension, "_EXTENSION_LOAD_ERROR", None)
    monkeypatch.setattr(extension, "_get_extension_path", _raise_missing_module)

    assert extension._load_library("_C_stable") is False
    assert isinstance(extension._EXTENSION_LOAD_ERROR, ImportError)
    assert "Could not find module '_C_stable'" in str(extension._EXTENSION_LOAD_ERROR)


def test_c_stable_load_error_not_hidden_by_later_library(monkeypatch):
    monkeypatch.setattr(extension, "_EXTENSION_LOAD_ERROR", None)
    monkeypatch.setattr(extension, "_get_extension_path", _raise_missing_module)

    assert extension._load_library("_C_stable") is False
    assert extension._load_library("image_stable") is False
    assert "Could not find module '_C_stable'" in str(extension._EXTENSION_LOAD_ERROR)


def test_assert_has_ops_includes_underlying_error(monkeypatch):
    load_error = ImportError("Could not find module '_C_stable' in /tmp/torchvision")
    monkeypatch.setattr(extension, "_has_ops", lambda: False)
    monkeypatch.setattr(extension, "_EXTENSION_LOAD_ERROR", load_error)

    with pytest.raises(RuntimeError, match=r"Couldn't load custom C\+\+ ops") as exc_info:
        extension._assert_has_ops()

    message = str(exc_info.value)
    assert "Underlying error: ImportError: Could not find module '_C_stable'" in message


def test_assert_has_ops_without_cached_error(monkeypatch):
    monkeypatch.setattr(extension, "_has_ops", lambda: False)
    monkeypatch.setattr(extension, "_EXTENSION_LOAD_ERROR", None)

    with pytest.raises(RuntimeError, match=r"Couldn't load custom C\+\+ ops") as exc_info:
        extension._assert_has_ops()

    assert "Underlying error:" not in str(exc_info.value)


def test_register_fake_is_guarded_when_ops_are_missing():
    # register_fake("torchvision::nms"|qnms) must sit inside an _has_ops() guard
    # so importing an unbuilt source tree does not raise "operator does not exist".
    source = inspect.getsource(meta_registrations)
    nms_idx = source.index('@torch.library.register_fake("torchvision::nms")')
    qnms_idx = source.index('@torch.library.register_fake("torchvision::qnms")')
    guard = "if torchvision.extension._has_ops():"
    nearest_guard = source.rfind(guard, 0, nms_idx)
    assert nearest_guard != -1
    assert nearest_guard < nms_idx < qnms_idx
    # The nms/qnms decorators share that same guard block.
    assert guard not in source[nms_idx:qnms_idx]
