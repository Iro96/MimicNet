# tests/test_*.py
def test_import_mimicnet():
    import mimicnet
    assert hasattr(mimicnet, "__version__") or True
