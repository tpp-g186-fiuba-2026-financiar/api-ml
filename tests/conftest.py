import pytest

from src import compare_snapshot, data
from src.config import settings


@pytest.fixture(autouse=True)
def _sin_copias_remotas(monkeypatch):
    """Los tests no bajan el snapshot ni el ledger de GitHub."""
    monkeypatch.setattr(settings, "compare_snapshot_url", None)
    monkeypatch.setattr(settings, "ledger_url", None)
    compare_snapshot._remote_cache.clear()


@pytest.fixture(autouse=True)
def _sin_cache_de_historicos():
    """Cada test arranca sin historicos cacheados ni colector marcado como caido."""
    data.reset_caches()
    yield
    data.reset_caches()
