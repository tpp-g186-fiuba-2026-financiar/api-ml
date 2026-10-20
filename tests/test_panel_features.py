import numpy as np
import pandas as pd
import pytest

from src.panel_datos import armar_panel
from src.panel_features import (
    FEATURES_ACCION,
    FLAGS,
    NIVEL,
    Normalizador,
    PreprocesoConfig,
    _racha,
    construir,
    contexto_macro,
    features_accion,
    particion,
    pesos_temporales,
    rank_gaussiano,
)

RUEDAS = pd.bdate_range("2024-01-01", periods=320)
# Dos financieras, dos petroleras y TECO2 sola en su sector.
TICKERS = ["GGAL", "BMA", "YPFD", "PAMP", "TECO2"]


def _ohlcv(seed: int, index: pd.DatetimeIndex = RUEDAS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(index))))
    abre = close * np.exp(rng.normal(0, 0.005, len(index)))
    volumen = rng.integers(500, 2000, len(index)).astype(float)
    return pd.DataFrame(
        {"open": abre, "high": np.maximum(close, abre) * 1.01,
         "low": np.minimum(close, abre) * 0.99, "close": close, "volume": volumen},
        index=index,
    )  # fmt: skip


def _histories() -> dict[str, pd.DataFrame]:
    h = {t: _ohlcv(i) for i, t in enumerate(TICKERS)}
    h["PAMP"] = h["PAMP"].iloc[100:]  # empieza tarde
    return h


def _panel(histories=None, **kw):
    return armar_panel(histories or _histories(), TICKERS, moneda="ars", **kw)


def test_rank_gaussiano() -> None:
    df = pd.DataFrame([[1.0, 2.0, 3.0], [5.0, np.nan, np.nan], [np.nan] * 3])
    r = rank_gaussiano(df)
    assert r.iloc[0, 0] == pytest.approx(-r.iloc[0, 2]) and r.iloc[0, 1] == pytest.approx(0)
    assert r.iloc[1, 0] == pytest.approx(0) and np.isnan(r.iloc[1, 1])
    assert r.iloc[2].isna().all()


def test_racha() -> None:
    r = pd.Series([np.nan, 0.1, 0.2, -0.1, -0.3, -0.2, 0.0, 0.4])
    assert _racha(r).tolist() == [0, 1, 2, -1, -2, -3, 0, 1]


def test_config_invalida() -> None:
    with pytest.raises(ValueError, match="target"):
        PreprocesoConfig(target="otro")
    with pytest.raises(ValueError, match="zona_neutral"):
        PreprocesoConfig(zona_neutral=-0.1)


@pytest.mark.parametrize("target", ["sin_media", "absoluto"])
def test_zona_neutral_saca_los_targets_chicos(target) -> None:
    panel = _panel()
    todo = construir(panel, PreprocesoConfig(target=target))
    zona = construir(panel, PreprocesoConfig(target=target, zona_neutral=0.5))
    # Solo achica la mascara del target; las features y el target no cambian.
    assert not (zona.mascara_target & ~todo.mascara_target).any()
    assert zona.mascara_target.sum() < todo.mascara_target.sum()
    assert (np.abs(zona.target[zona.mascara_target]) >= 0.5).all()
    sacadas = todo.mascara_target & ~zona.mascara_target
    assert (np.abs(todo.target[sacadas]) < 0.5).all()
    np.testing.assert_array_equal(zona.valores, todo.valores)
    np.testing.assert_array_equal(zona.mascara, todo.mascara)


def test_construir_formas_y_mascaras() -> None:
    panel = _panel()
    fp = construir(panel)
    assert fp.nombres == FEATURES_ACCION + NIVEL + FLAGS
    assert fp.valores.shape == (len(RUEDAS), len(TICKERS), len(fp.nombres))
    assert np.isfinite(fp.valores).all()
    assert fp.sectores == ["financiero", "financiero", "petroleo", "petroleo", "telecom"]
    # La mascara de features es mas estricta que la del panel (pide vol conocida).
    assert not (fp.mascara & ~panel.mascara.to_numpy()).any()
    assert not fp.mascara[:5].any() and fp.mascara[50:, 0].all()
    assert not fp.mascara[:100, TICKERS.index("PAMP")].any()
    # El target pide futuro: las ultimas `horizonte` fechas no tienen.
    assert not fp.mascara_target[-5:].any()
    assert not (fp.mascara_target & ~fp.mascara).any()
    # Sin la media del dia, el target promedia 0 cada dia.
    con_target = fp.mascara_target.any(axis=1)
    medias = np.nanmean(np.where(fp.mascara_target, fp.target, np.nan)[con_target], axis=1)
    assert np.abs(medias).max() < 1e-9
    assert np.nanmax(np.abs(fp.target)) <= 2 * PreprocesoConfig().clip_target
    # Ranks: features rankeadas centradas en 0 cada dia.
    ret5 = fp.valores[:, :, fp.nombres.index("ret_5")]
    dia = 200
    assert ret5[dia][fp.mascara[dia]].sum() == pytest.approx(0, abs=1e-9)


def test_retorno_crudo_y_target_crudo() -> None:
    panel = _panel()
    fp = construir(panel, PreprocesoConfig(target="crudo"))
    close = panel["close"].to_numpy()
    t, n = 150, 0
    assert fp.retorno[t, n] == pytest.approx(np.log(close[t + 5, n] / close[t, n]))
    np.testing.assert_allclose(fp.target, fp.retorno, equal_nan=True)


def test_target_absoluto_conserva_la_direccion() -> None:
    panel = _panel()
    sin_media = construir(panel, PreprocesoConfig())
    absoluto = construir(panel, PreprocesoConfig(target="absoluto"))
    m = absoluto.mascara_target
    assert np.array_equal(sin_media.mascara_target, m)
    assert np.array_equal(np.sign(absoluto.target[m]), np.sign(absoluto.retorno[m]))
    assert np.abs(absoluto.target[m]).max() <= PreprocesoConfig().clip_target
    # Restarle la media del dia da el target "sin_media".
    centrado = np.where(m, absoluto.target, np.nan)
    centrado = centrado - np.nanmean(centrado, axis=1, keepdims=True)
    np.testing.assert_allclose(centrado[m], sin_media.target[m])


def test_target_residual_saca_el_mercado() -> None:
    panel = _panel()
    sin_media = construir(panel, PreprocesoConfig())
    residual = construir(panel, PreprocesoConfig(target="residual"))
    assert np.array_equal(sin_media.mascara_target, residual.mascara_target)
    assert not np.allclose(sin_media.target, residual.target, equal_nan=True)
    assert np.isfinite(residual.target[residual.mascara_target]).all()


def test_features_de_reversion_sin_rankear() -> None:
    panel = _panel()
    config = PreprocesoConfig(normalizar_vol=False, rankear=False)
    f = features_accion(panel, config)
    dia = RUEDAS[200]
    ret5 = np.log(panel["close"]).diff(5).loc[dia]
    assert f["ret_5"].loc[dia].to_numpy() == pytest.approx(ret5.to_numpy())
    assert f["rel_panel_5"].loc[dia].sum() == pytest.approx(0, abs=1e-12)
    financieras = f["rel_sector_5"].loc[dia, ["GGAL", "BMA"]]
    assert financieras.sum() == pytest.approx(0, abs=1e-12)
    assert f["rel_sector_5"].loc[dia, "TECO2"] == pytest.approx(0)  # sola en su sector
    assert f["beta_60"].loc[dia].notna().all() and f["corr_60"].loc[dia].between(-1, 1).all()
    # Sin normalizar ni rankear, construir deja la feature cruda.
    fp = construir(panel, config)
    i = fp.nombres.index("ret_5")
    assert fp.valores[200, :, i] == pytest.approx(ret5.to_numpy())


def test_normalizar_por_vol_recorta() -> None:
    f = features_accion(_panel(), PreprocesoConfig(clip_features=1.0))
    for nombre in ("ret_1", "ret_60", "rel_panel_5", "gap"):
        assert f[nombre].abs().max().max() <= 1.0


@pytest.mark.parametrize("target", ["sin_media", "residual", "absoluto", "crudo"])
def test_no_mira_el_futuro(target) -> None:
    """Features y mascara de la fecha t no cambian si se borran los datos
    posteriores a t; el target de t solo usa datos hasta t + horizonte."""
    histories = _histories()
    config = PreprocesoConfig(target=target)
    completo = construir(_panel(histories), config)
    for i in (120, 200, 300):
        corte = RUEDAS[i]
        hasta = {t: df[df.index <= corte] for t, df in histories.items()}
        parcial = construir(_panel(hasta), config)
        n = len(parcial.fechas)
        np.testing.assert_allclose(parcial.valores, completo.valores[:n])
        assert np.array_equal(parcial.mascara, completo.mascara[:n])
        # Con datos hasta t, el target de t - horizonte ya es el definitivo.
        h = config.horizonte
        np.testing.assert_allclose(
            parcial.target[: n - h], completo.target[: n - h], equal_nan=True
        )


def test_tabla() -> None:
    fp = construir(_panel())
    tabla = fp.tabla()
    assert len(tabla) == fp.mascara.sum()
    assert list(tabla.columns[:2]) == ["fecha", "ticker"]
    assert {"target", "retorno", *fp.nombres} <= set(tabla.columns)


def test_contexto_macro() -> None:
    dias = pd.date_range(RUEDAS[0] - pd.Timedelta(days=400), RUEDAS[-1], freq="D")
    rng = np.random.default_rng(0)
    macro = pd.DataFrame(
        {"ccl": np.linspace(1000, 1500, len(dias)), "oficial": np.linspace(800, 1400, len(dias)),
         "riesgo_pais": 600 + rng.normal(0, 10, len(dias)), "badlar": 30.0},
        index=dias,
    )  # fmt: skip
    ctx = contexto_macro(macro, RUEDAS)
    assert ctx.index.equals(RUEDAS)
    assert {"a_brecha", "a_rp_z", "e_days_to_el"} <= set(ctx.columns)
    assert ctx["a_brecha"].iloc[-1] == pytest.approx(1500 / 1400 - 1)


def test_normalizador() -> None:
    rng = np.random.default_rng(1)
    x = np.stack([rng.normal(5, 2, (50, 4)), np.full((50, 4), 3.0)], axis=-1)
    mascara = np.zeros((50, 4), dtype=bool)
    mascara[:30] = True
    norm = Normalizador().fit(x, mascara)
    z = norm.transform(x)
    assert np.nanmean(z[mascara][:, 0]) == pytest.approx(0, abs=1e-9)
    assert np.nanstd(z[mascara][:, 0]) == pytest.approx(1)
    assert (z[..., 1] == 0).all()  # feature constante: no divide por 0
    assert Normalizador().fit(x).media.shape == (2,)
    with pytest.raises(ValueError, match="ajustado"):
        Normalizador().transform(x)


def test_particion_con_gap() -> None:
    train, val = particion(RUEDAS[:100], val_fraction=0.2, gap=5)
    assert val[0] - train[-1] == 6 and len(val) == 20 and train[0] == 0
    with pytest.raises(ValueError, match="pocas fechas"):
        particion(RUEDAS[:4], val_fraction=0.5, gap=5)


def test_pesos_temporales() -> None:
    fechas = pd.DatetimeIndex(["2020-01-01", "2022-01-01", "2024-01-01"])
    pesos = pesos_temporales(fechas, vida_media_anios=2.0)
    assert pesos.mean() == pytest.approx(1)
    assert pesos[2] / pesos[1] == pytest.approx(2, rel=1e-3)
    assert (pesos_temporales(fechas, None) == 1).all()
