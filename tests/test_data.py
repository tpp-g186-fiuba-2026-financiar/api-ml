"""Tests del mapeo de simbolos de Yahoo (sin red)."""

from src.data import to_yahoo_symbol


def test_byma_ticker_gets_suffix() -> None:
    assert to_yahoo_symbol("GGAL") == "GGAL.BA"


def test_symbol_with_suffix_is_respected() -> None:
    assert to_yahoo_symbol("AAPL") == "AAPL.BA"  # sin "." ni "=": lleva sufijo igual
    assert to_yahoo_symbol("BRK.B") == "BRK.B"
    assert to_yahoo_symbol("GC=F") == "GC=F"


def test_commodities_resolve_to_yahoo_futures_not_byma() -> None:
    """GOLD/OIL son commodities de data-colector, no acciones -- sin esto Yahoo da 404."""
    assert to_yahoo_symbol("GOLD") == "GC=F"
    assert to_yahoo_symbol("OIL") == "CL=F"


def test_commodity_mapping_is_case_insensitive() -> None:
    assert to_yahoo_symbol("gold") == "GC=F"
