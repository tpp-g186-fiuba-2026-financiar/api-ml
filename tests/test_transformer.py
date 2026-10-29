"""Tests del pipeline Transformer con datos sinteticos (sin red)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src import transformer
from src.errors import DataUnavailableError, NotEnoughDataError
from src.transformer import (
    CONFIG_ORIGINAL,
    TransformerTrainConfig,
    TransformerTrendModel,
    _bloques_walk_forward,
)


def _synthetic_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _small_config(**overrides) -> TransformerTrainConfig:
    base = dict(
        window=10,
        horizon=5,
        d_model=4,
        nhead=2,
        num_layers=1,
        dim_feedforward=8,
        dropout=0.0,
        epochs=3,
        batch_size=16,
        patience=1,
        warmup_epochs=1,
        # En pesos y sin riesgo pais, para no depender del data-colector; los
        # tests de dolares / riesgo pais pasan un proveedor falso.
        moneda="ars",
        riesgo_pais=False,
    )
    base.update(overrides)
    return TransformerTrainConfig(**base)


def test_train_predict_save_load(tmp_path) -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    model = TransformerTrendModel(config=_small_config())
    assert model.version == "untrained"

    metrics = model.fit(histories)
    assert "val_directional_accuracy" in metrics
    assert model.version.startswith("transformer-")

    result = model.predict_df(histories["SYN0"])
    assert result["signal"] in {"alza", "baja", "neutral"}
    assert result["condition"] in {"sobrecompra", "sobreventa", "neutral", "indeterminado"}
    assert "predicted_close" in result

    path = tmp_path / "transformer.pt"
    model.save(path)
    reloaded = TransformerTrendModel.load(path)
    reloaded_result = reloaded.predict_df(histories["SYN0"])
    assert reloaded_result["predicted_close"] == pytest.approx(result["predicted_close"], rel=1e-4)


def test_fit_without_validation_split() -> None:
    """val_fraction=0 deja Xva vacio: cubre la rama sin early stopping."""
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    config = _small_config(val_fraction=0.0, epochs=2)
    model = TransformerTrendModel(config=config)
    metrics = model.fit(histories)
    assert metrics == {}


def test_fit_raises_without_enough_data() -> None:
    histories = {"SYN0": _synthetic_ohlcv(n=5)}
    model = TransformerTrendModel(config=_small_config())
    with pytest.raises(NotEnoughDataError):
        model.fit(histories)


def test_predict_requires_enough_data() -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    model = TransformerTrendModel(config=_small_config())
    model.fit(histories)
    with pytest.raises(NotEnoughDataError):
        model.predict_df(_synthetic_ohlcv(n=5))


def test_config_rejects_incompatible_d_model_and_nhead() -> None:
    with pytest.raises(ValueError):
        TransformerTrainConfig(d_model=5, nhead=2)


def test_variantes_relativo_panel_huber_ic_ensamble(tmp_path) -> None:
    """Target relativo, universo panel, Huber, recorte, gap, seleccion por IC y ensamble."""
    panel = ["GGAL", "BMA", "YPFD", "PAMP"]
    histories = {t: _synthetic_ohlcv(seed=i) for i, t in enumerate(panel)}
    histories["FUERA"] = _synthetic_ohlcv(seed=9)
    config = _small_config(
        **CONFIG_ORIGINAL,
        target="relativo",
        universo="panel",
        loss="huber",
        clip_target=3.0,
        gap=5,
        seleccion="ic",
        n_semillas=2,
        banda_cuantil=0.5,
    )
    model = TransformerTrendModel(config=config)
    metrics = model.fit(histories)

    assert set(model.tickers) == set(panel)
    assert "val_ic" in metrics
    assert len(model._nets) == 2
    assert model.config.neutral_band == pytest.approx(metrics["neutral_band"], abs=1e-6)
    assert model.config.neutral_band != 0.01
    # El target relativo no tiene deriva: la media de los excesos es ~0.
    assert abs(model.target_mean) < 1e-3

    result = model.predict_df(histories["GGAL"])
    path = tmp_path / "transformer.pt"
    model.save(path)
    reloaded = TransformerTrendModel.load(path)
    assert len(reloaded._nets) == 2
    assert reloaded.config.target == "relativo"
    assert reloaded.predict_df(histories["GGAL"])["predicted_close"] == pytest.approx(
        result["predicted_close"], rel=1e-4
    )


def test_load_de_artefacto_de_una_sola_red(tmp_path) -> None:
    """Artefactos viejos: solo "state_dict" y un config sin las opciones nuevas."""
    import torch

    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    model = TransformerTrendModel(config=_small_config(**CONFIG_ORIGINAL))
    model.fit(histories)
    path = tmp_path / "viejo.pt"
    model.save(path)
    blob = torch.load(path, weights_only=False)
    del blob["state_dicts"]
    claves = ("target", "universo", "loss", "clip_target", "seleccion", "gap", "n_semillas")
    for clave in (*claves, *CONFIG_ORIGINAL):
        del blob["config"][clave]
    torch.save(blob, path)

    reloaded = TransformerTrendModel.load(path)
    assert reloaded.config.target == "absoluto"
    assert reloaded.config.objetivo == "regresion"
    assert reloaded._net is not None
    assert reloaded.predict_df(histories["SYN0"])["predicted_close"] == pytest.approx(
        model.predict_df(histories["SYN0"])["predicted_close"], rel=1e-4
    )


def test_relativo_sin_fechas_comunes_no_tiene_datos() -> None:
    """Con menos de 3 tickers por fecha no hay promedio de mercado."""
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    model = TransformerTrendModel(config=_small_config(target="relativo"))
    with pytest.raises(NotEnoughDataError):
        model.fit(histories)


@pytest.mark.parametrize(
    "cambio",
    [
        {"target": "otro"},
        {"universo": "otro"},
        {"loss": "otro"},
        {"seleccion": "otro"},
        {"n_semillas": 0},
        {"banda_cuantil": 1.5},
    ],
)
def test_config_rechaza_opciones_invalidas(cambio) -> None:
    with pytest.raises(ValueError):
        TransformerTrainConfig(**cambio)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sin GPU")
def test_entrena_en_gpu_y_deja_las_redes_en_cpu() -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    model = TransformerTrendModel(config=_small_config(seleccion="ic"), dispositivo="cuda")
    model.fit(histories)
    assert next(model._net.parameters()).device.type == "cpu"
    assert model.predict_df(histories["SYN0"])["signal"] in {"alza", "baja", "neutral"}


def _panel(n: int = 200) -> dict[str, pd.DataFrame]:
    from src.black_litterman.black_litterman import PANEL_LIDER_TICKERS

    return {s: _synthetic_ohlcv(n=n, seed=i) for i, s in enumerate(PANEL_LIDER_TICKERS[:6])}


def test_features_panel_y_escala_vol() -> None:
    from src.transformer import _FEATURES_PANEL, contexto_panel, n_features

    cfg = _small_config(features_panel=True, escala_vol=True)
    assert n_features(cfg) == 8 + len(_FEATURES_PANEL)
    histories = _panel()
    model = TransformerTrendModel(config=cfg)
    model.fit(histories)
    assert model._net.input_proj.in_features == n_features(cfg)

    resultados = model.predict_panel(histories)
    assert set(resultados) == set(histories)
    assert all(r["signal"] in {"alza", "baja", "neutral"} for r in resultados.values())
    # Sin el panel del dia no se puede predecir.
    with pytest.raises(NotEnoughDataError, match="panel"):
        model.predict_df(histories["ALUA"])
    # Con menos de 5 tickers del panel no hay contexto.
    assert contexto_panel(dict(list(histories.items())[:4])) is None
    with pytest.raises(NotEnoughDataError):
        TransformerTrendModel(config=cfg).fit(dict(list(histories.items())[:4]))


def test_features_panel_no_miran_el_futuro() -> None:
    """Las features de una rueda no cambian si se agregan ruedas posteriores."""
    from src.transformer import _features, contexto_panel

    cfg = _small_config(features_panel=True, escala_vol=True)
    completo = _panel()
    cortado = {s: df.iloc[:150] for s, df in completo.items()}
    f_completo, _, e_completo = _features(completo["ALUA"], cfg, contexto_panel(completo))
    f_cortado, _, e_cortado = _features(cortado["ALUA"], cfg, contexto_panel(cortado))
    np.testing.assert_allclose(f_completo[:149], f_cortado)
    np.testing.assert_allclose(e_completo[:149], e_cortado)


def test_predict_panel_coincide_con_predict_df_y_omite_cortos() -> None:
    histories = _panel()
    model = TransformerTrendModel(config=_small_config(escala_vol=True))
    model.fit(histories)
    dfs = {**histories, "CORTO": _synthetic_ohlcv(n=8)}
    resultados = model.predict_panel(dfs)
    assert "CORTO" not in resultados
    individual = model.predict_df(histories["BBAR"])
    assert resultados["BBAR"]["predicted_close"] == pytest.approx(individual["predicted_close"])
    assert model.predict_panel({"CORTO": dfs["CORTO"]}) == {}
    with pytest.raises(NotEnoughDataError):
        TransformerTrendModel().predict_panel(histories)
    plano = _synthetic_ohlcv()
    plano["close"] = 100.0
    with pytest.raises(NotEnoughDataError, match="volatilidad"):
        model.predict_df(plano)


def test_banda_cuantil_no_se_combina_con_escala_vol() -> None:
    with pytest.raises(ValueError, match="escala_vol"):
        TransformerTrainConfig(escala_vol=True, banda_cuantil=0.5)


def test_columnas_elige_las_features_de_la_red() -> None:
    from src.transformer import n_features, nombres_features

    cfg = _small_config(features_panel=True, target="relativo", columnas=["ret5_rel", "rsi_norm"])
    assert nombres_features(cfg) == ["ret5_rel", "rsi_norm"] and n_features(cfg) == 2
    model = TransformerTrendModel(config=cfg)
    model.fit(_panel())
    assert model._net.input_proj.in_features == 2
    assert len(model.predict_panel(_panel())) == 6
    with pytest.raises(ValueError, match="columnas"):
        TransformerTrainConfig(columnas=["ret5_rel"])  # sin features_panel no existe
    with pytest.raises(ValueError, match="columnas"):
        TransformerTrainConfig(columnas=[])


def test_bloques_walk_forward_comunes_a_todos_los_tickers() -> None:
    """Tickers con historias de distinto largo comparten los bloques, con gap."""
    fechas = pd.date_range("2024-01-01", periods=100, freq="B").to_numpy()
    vacio = np.zeros(0)
    por_ticker = {"LARGO": (vacio, vacio, fechas), "CORTO": (vacio, vacio, fechas[60:])}
    bloques = _bloques_walk_forward(por_ticker, val_fraction=0.3, folds=3, gap=5)

    assert [(b[0], b[1]) for b in bloques] == [
        (fechas[65], fechas[70]),
        (fechas[75], fechas[80]),
        (fechas[85], fechas[90]),
    ]
    # Bloques consecutivos que cubren hasta la ultima fecha.
    assert bloques[0][2] == bloques[1][1] and bloques[1][2] == bloques[2][1]
    assert bloques[-1][2] > fechas[-1]


def test_bloques_walk_forward_sin_validacion_y_sin_datos() -> None:
    fechas = pd.date_range("2024-01-01", periods=20, freq="B").to_numpy()
    vacio = np.zeros(0)
    assert _bloques_walk_forward({"A": (vacio, vacio, fechas)}, 0.0, folds=3, gap=5) == []
    # Menos fechas de validacion que bloques: un bloque por fecha.
    assert len(_bloques_walk_forward({"A": (vacio, vacio, fechas)}, 0.1, folds=3, gap=5)) == 2
    with pytest.raises(NotEnoughDataError):
        _bloques_walk_forward({"A": (vacio, vacio, fechas)}, 0.5, folds=1, gap=10)


def test_gap_por_defecto_es_el_horizonte_y_no_acepta_negativos() -> None:
    assert TransformerTrainConfig().gap is None
    with pytest.raises(ValueError):
        TransformerTrainConfig(gap=-1)


def test_clasificacion_por_defecto_retorno_acotado_por_el_movimiento_tipico(tmp_path) -> None:
    """La red da el logit de "sube"; el retorno esperado es (2p - 1) * movimiento tipico."""
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    model = TransformerTrendModel(config=_small_config())
    assert model.config.objetivo == "clasificacion"
    metrics = model.fit(histories)
    assert "val_acierto_balanceado" in metrics
    assert model.target_mean == 0.0
    assert model.target_std > 0

    x = torch.randn(8, model.config.window, len(model.feature_mean))
    retornos = model._salida(x)
    assert np.all(np.abs(retornos) <= model.target_std)

    result = model.predict_df(histories["SYN0"])
    path = tmp_path / "clasificacion.pt"
    model.save(path)
    reloaded = TransformerTrendModel.load(path)
    assert reloaded.config.objetivo == "clasificacion"
    assert reloaded.predict_df(histories["SYN0"])["predicted_close"] == pytest.approx(
        result["predicted_close"], rel=1e-4
    )


def test_clasificacion_con_probabilidad_media_es_neutral() -> None:
    """Logit 0 (p = 0.5) => retorno esperado 0."""
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    model = TransformerTrendModel(config=_small_config())
    model.fit(histories)
    model._forward = lambda x: np.zeros(len(x), dtype=np.float32)
    assert model.predict_df(histories["SYN0"])["signal"] == "neutral"


def test_clasificacion_rechaza_opciones_de_regresion() -> None:
    with pytest.raises(ValueError):
        TransformerTrainConfig(loss="huber")
    with pytest.raises(ValueError):
        TransformerTrainConfig(clip_target=3.0)
    with pytest.raises(ValueError):
        TransformerTrainConfig(objetivo="otro")
    assert TransformerTrainConfig(objetivo="regresion", loss="huber").loss == "huber"


def test_walk_forward_entrena_un_bloque_por_fold_y_al_final_con_todo(monkeypatch) -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    llamadas = []
    original = TransformerTrendModel._entrenar_ensamble

    def espia(self, X, y, Xva=None, yva=None, fva=None, epocas=None):
        llamadas.append((len(X), epocas))
        return original(self, X, y, Xva, yva, fva, epocas)

    monkeypatch.setattr(TransformerTrendModel, "_entrenar_ensamble", espia)
    model = TransformerTrendModel(config=_small_config(folds=3))
    metrics = model.fit(histories)

    assert metrics["folds"] == 3
    assert len(llamadas) == 4
    # Cada bloque entrena con mas fechas que el anterior; el final, con todas.
    tamanios = [n for n, _ in llamadas]
    assert tamanios == sorted(tamanios) and tamanios[-1] > tamanios[-2]
    assert llamadas[-1][1] == metrics["epocas"]
    assert all(e is None for _, e in llamadas[:-1])

    llamadas.clear()
    TransformerTrendModel(config=_small_config(folds=2, reentrenar_completo=False)).fit(histories)
    assert len(llamadas) == 2


def _ccl(n: int = 200, seed: int = 0) -> pd.Series:
    """CCL que sube ~0,3% por rueda (devaluacion)."""
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(1000 * np.exp(0.003 * np.arange(n)), index=index)


def _proveedor_falso(ccl: pd.Series | None, riesgo_pais: bool = True):
    llamadas = []

    def proveedor(hasta):
        llamadas.append(hasta)
        macro = None
        if riesgo_pais:
            dias = pd.date_range("2022-12-01", hasta, freq="D")
            macro = pd.DataFrame({"riesgo_pais": np.linspace(2000, 800, len(dias))}, index=dias)
        return (None if ccl is None else ccl[ccl.index <= hasta]), macro

    proveedor.llamadas = llamadas
    return proveedor


def test_dolares_y_riesgo_pais_con_proveedor(tmp_path) -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    proveedor = _proveedor_falso(_ccl())
    config = _small_config(moneda="usd", riesgo_pais=True)
    model = TransformerTrendModel(config=config, proveedor=proveedor)
    model.fit(histories)
    assert model._net.input_proj.in_features == len(transformer.FEATURE_NAMES) + 1

    preparado = model._preparar(histories["SYN0"])
    esperado = histories["SYN0"]["close"] / _ccl()
    assert preparado["close"].to_numpy() == pytest.approx(esperado.to_numpy())
    assert preparado["riesgo_pais"].notna().all()

    # El precio proyectado sigue en pesos (a CCL constante).
    result = model.predict_df(histories["SYN0"])
    last = float(histories["SYN0"]["close"].iloc[-1])
    assert result["last_close"] == pytest.approx(last, rel=1e-4)
    assert 0.5 * last < result["predicted_close"] < 1.5 * last

    path = tmp_path / "usd.pt"
    model.save(path)
    reloaded = TransformerTrendModel.load(path, proveedor=proveedor)
    assert reloaded.config.moneda == "usd"
    assert reloaded.predict_df(histories["SYN0"])["predicted_close"] == pytest.approx(
        result["predicted_close"], rel=1e-4
    )


def test_dolares_sin_ccl_no_entrena() -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(2)}
    config = _small_config(moneda="usd")
    model = TransformerTrendModel(config=config, proveedor=_proveedor_falso(None, False))
    with pytest.raises(DataUnavailableError):
        model.fit(histories)


def test_riesgo_pais_sin_dato_queda_en_cero() -> None:
    df = _synthetic_ohlcv()
    model = TransformerTrendModel(
        config=_small_config(riesgo_pais=True), proveedor=_proveedor_falso(None, False)
    )
    assert model._preparar(df)["riesgo_pais"].isna().all()
    feats, _, _ = transformer._features(model._preparar(df), model.config)
    assert np.all(feats[:, -1] == 0)


def test_limpieza_ajusta_un_split_al_entrenar_y_al_predecir(tmp_path) -> None:
    histories = {f"SYN{i}": _synthetic_ohlcv(seed=i) for i in range(3)}
    con_split = histories["SYN0"].copy()
    fecha = con_split.index[150]
    for col in ("open", "high", "low", "close"):
        con_split.loc[con_split.index >= fecha, col] /= 10
    con_split.iloc[10, con_split.columns.get_loc("close")] = 0.0  # precio invalido
    histories["SYN0"] = con_split

    model = TransformerTrendModel(config=_small_config())
    model.fit(histories)
    limpio = model._preparar(con_split)
    retornos = np.log(limpio["close"]).diff()
    assert abs(retornos.loc[fecha]) < 0.1
    assert len(limpio) == len(con_split) - 1
    assert model.mercado is not None

    path = tmp_path / "limpio.pt"
    model.save(path)
    assert TransformerTrendModel.load(path).mercado is not None

    sin_limpiar = TransformerTrendModel(config=_small_config(limpiar=False))
    assert sin_limpiar._preparar(con_split) is con_split


def test_proveedor_collector_baja_las_series_una_vez_por_dia(monkeypatch) -> None:
    pedidas = []
    index = pd.date_range("2024-01-01", periods=30, freq="D")

    def fetch(ruta, serie):
        pedidas.append(serie)
        return pd.Series(np.arange(1.0, 31.0), index=index)

    monkeypatch.setattr(transformer, "fetch_collector_series", fetch)
    monkeypatch.setattr(transformer, "_proveedor_del_dia", None)
    proveedor = transformer.proveedor_collector()
    assert transformer.proveedor_collector() is proveedor
    assert pedidas == ["CCL", "RIESGO_PAIS"]

    ccl, macro = proveedor(index[9])
    assert ccl.index[-1] == index[9]
    # El riesgo pais tiene 2 dias de retraso de publicacion.
    assert macro["riesgo_pais"].iloc[-1] == 8.0


def test_config_rechaza_folds_y_moneda_invalidos() -> None:
    with pytest.raises(ValueError):
        TransformerTrainConfig(folds=0)
    with pytest.raises(ValueError):
        TransformerTrainConfig(moneda="eur")
    assert TransformerTrainConfig().necesita_proveedor
    assert not TransformerTrainConfig(**CONFIG_ORIGINAL).necesita_proveedor


def test_en_dolares_excluye_commodities_y_especies_d() -> None:
    assert transformer.cotiza_en_dolares("GOLD") and transformer.cotiza_en_dolares("aluad")
    assert not transformer.cotiza_en_dolares("YPFD") and not transformer.cotiza_en_dolares("GGAL")

    histories = {s: _synthetic_ohlcv(seed=i) for i, s in enumerate(["GGAL", "BMA", "OIL", "ALUAD"])}
    # Los futuros vienen con otra hora: las fechas se comparan por dia.
    histories["OIL"].index = histories["OIL"].index + pd.Timedelta(hours=4)
    histories["GGAL"].index = histories["GGAL"].index + pd.Timedelta(hours=14)
    model = TransformerTrendModel(
        config=_small_config(moneda="usd"), proveedor=_proveedor_falso(_ccl(), False)
    )
    model.fit(histories)
    assert sorted(model.tickers) == ["BMA", "GGAL"]
    assert not model.acepta("OIL") and model.acepta("GGAL")
    assert TransformerTrendModel(config=_small_config()).acepta("OIL")  # en pesos, todos

    ventanas = transformer._ventanas_con_fecha(histories["OIL"], model.config)
    assert (pd.DatetimeIndex(ventanas[2]) == pd.DatetimeIndex(ventanas[2]).normalize()).all()
