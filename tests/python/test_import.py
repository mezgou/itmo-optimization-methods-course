from __future__ import annotations

import optlib


def test_import_and_version() -> None:
    assert optlib.__version__ == "1.0.0"
    assert optlib.Version() == "1.0.0"


def test_extension_smoke_function() -> None:
    assert optlib.Add(2.0, 3.0) == 5.0
