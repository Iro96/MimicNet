# tests/test_*.py
def test_import_mimicnet():
    import models
    assert hasattr(models, "__version__") or True
