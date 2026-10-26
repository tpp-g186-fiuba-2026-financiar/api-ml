import pytest

from src import compare_snapshot
from src.config import settings


@pytest.fixture(autouse=True)
def _sin_copias_remotas(monkeypatch):
    """Los tests no bajan el snapshot ni el ledger de GitHub."""
    monkeypatch.setattr(settings, "compare_snapshot_url", None)
    monkeypatch.setattr(settings, "ledger_url", None)
    compare_snapshot._remote_cache.clear()
