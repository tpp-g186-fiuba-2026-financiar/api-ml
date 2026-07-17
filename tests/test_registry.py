"""Tests del TrendRegistry / TrendService (sin red)."""

from src.errors import StaleArtifactError
from src.registry import TrendService


def test_load_tolerates_incompatible_artifact(tmp_path) -> None:
    """Un artefacto que no matchea el feature set actual no debe tirar abajo el resto."""
    path = tmp_path / "fake.pt"
    path.write_bytes(b"no importa el contenido, solo que el archivo exista")

    def _raising_loader(_path):
        raise StaleArtifactError("artefacto viejo, hay que reentrenar")

    service = TrendService("lstm", str(path), _raising_loader, history_days=750)
    service.load()  # no debe propagar la excepcion

    assert service.is_loaded is False
    assert service.version == "unloaded"


def test_load_skips_missing_artifact(tmp_path) -> None:
    path = tmp_path / "missing.pt"
    service = TrendService("lstm", str(path), lambda p: None, history_days=750)
    service.load()
    assert service.is_loaded is False
