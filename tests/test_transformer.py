"""Tests del pipeline Transformer con datos sinteticos (sin red)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.errors import NotEnoughDataError
from src.transformer import TransformerTrainConfig, TransformerTrendModel


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
    model = TransformerTrendModel(config=_small_config())
    model.fit(histories)
    path = tmp_path / "viejo.pt"
    model.save(path)
    blob = torch.load(path, weights_only=False)
    del blob["state_dicts"]
    for clave in ("target", "universo", "loss", "clip_target", "seleccion", "gap", "n_semillas"):
        del blob["config"][clave]
    torch.save(blob, path)

    reloaded = TransformerTrendModel.load(path)
    assert reloaded.config.target == "absoluto"
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
