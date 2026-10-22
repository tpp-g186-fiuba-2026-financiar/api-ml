import dataclasses

import numpy as np
import pytest
import torch

from src.errors import ApiMlError, NotEnoughDataError
from src.panel.gbm import features_y_contexto
from src.panel.transformer import PanelTransformerModel
from src.panel.transformer_v2 import CONFIG_ANTERIOR, TransformerV2Config, TransformerV2Model
from src.trend_common import acierto_balanceado
from tests.panel.test_transformer import (
    RUEDAS,
    TICKERS,
    _historias_con_reversion,
    _histories,
    _ohlcv,
    _proveedor,
)

# Los tests de la maquinaria de direccion usan la formulacion absoluta (la anterior
# a Optuna), que tiene ``val_direccion`` y el ensamble con la reversion.
CHICO = dict(
    ventana=5, d_model=16, nhead=2, dim_feedforward=32, epocas=3, batch_dias=32, n_semillas=1,
    target="absoluto", seleccion="direccion", perdida="huber", capas_temporales=1,
)  # fmt: skip


def _modelo(**extra) -> TransformerV2Model:
    return TransformerV2Model(TransformerV2Config(**(CHICO | extra)), proveedor=_proveedor)


def test_defaults_son_los_de_optuna() -> None:
    cfg = TransformerV2Config()
    assert (cfg.moneda, cfg.target, cfg.seleccion, cfg.perdida) == (
        "usd", "sin_media", "perdida", "bce"
    )  # fmt: skip
    assert (cfg.d_model, cfg.nhead, cfg.capas_temporales, cfg.ventana) == (16, 4, 2, 1)
    # Los valores anteriores siguen disponibles para los planes viejos de la busqueda.
    anterior = TransformerV2Config(**CONFIG_ANTERIOR)
    assert (anterior.target, anterior.weight_decay) == ("absoluto", 0.3)
    assert cfg.limpiar_saltos and cfg.reentrenar_completo and cfg.cobertura_senal is None
    assert TransformerV2Config(perdida="bce").perdida == "bce"
    for extra, match in (({"cobertura_senal": 0.0}, "cobertura"), ({"perdida": "mse"}, "bce")):
        with pytest.raises(ValueError, match=match):
            TransformerV2Config(**extra)


@pytest.mark.parametrize("reentrenar", [True, False])
def test_fit_y_predict_panel(reentrenar) -> None:
    modelo = _modelo(reentrenar_completo=reentrenar)
    info = modelo.fit(_histories())
    n_dias = len(RUEDAS) - modelo.config.ventana  # aprox: sin las ultimas ni la ventana
    # Con reentreno se entrena con todos los dias (train + gap + validacion).
    assert (info["dias_train"] > n_dias * 0.9) == reentrenar
    assert len(modelo._redes) == 1 and info["epoca_mejor"][0] >= 1
    resultados = modelo.predict_panel(_histories())
    assert set(resultados) == set(TICKERS)
    for r in resultados.values():
        retorno = np.log(r["predicted_close"] / r["last_close"])
        assert r["signal"] == ("alza" if retorno > 0 else "baja" if retorno < 0 else "neutral")
    assert modelo.version == "transformer-v2-usd"


def test_cobertura_senal_solo_marca_las_predicciones_fuertes() -> None:
    modelo = _modelo(cobertura_senal=0.2)
    info = modelo.fit(_histories())
    assert info["umbral_senal"] > 0
    senales, fuerzas = [], []
    for corte in range(450, 600, 5):
        dia = modelo.predict_panel({t: df.iloc[:corte] for t, df in _histories().items()})
        for r in dia.values():
            fuerza = abs(np.log(r["predicted_close"] / r["last_close"]))
            fuerzas.append(fuerza)
            senales.append(r["signal"])
            assert (r["signal"] != "neutral") == (fuerza > info["umbral_senal"])
    # El umbral se calibro en entrenamiento: la cobertura fuera de muestra es parecida.
    assert 0.05 < np.mean([s != "neutral" for s in senales]) < 0.5


def test_centrar_senal_deja_la_mediana_de_entrenamiento_en_cero() -> None:
    """Centrada, la mitad de las predicciones de entrenamiento es "alza" y la
    otra mitad "baja", aunque la red tenga sesgo."""
    modelo = _modelo(centrar_senal=True, peso_reversion=0.3)
    modelo.fit(_histories())
    assert "centro_senal" in modelo.metrics
    fp, ctx = features_y_contexto(_histories(), modelo.config, _proveedor)
    dias = modelo._dias_muestra(fp)
    # Los mismos tensores que en el reentreno (normalizadores con toda la historia).
    datos = modelo._datos(fp, ctx, hasta=len(fp.fechas))
    score, reversion, _vol = modelo._scores_entrenamiento(fp, datos, dias)
    ajustado = modelo._combinar(score, reversion) - modelo.metrics["centro_senal"]
    assert abs(np.median(ajustado)) < 1e-6
    assert 0.45 < np.mean(ajustado > 0) < 0.55


def test_peso_reversion_uno_es_la_regla_de_reversion() -> None:
    """Con peso 1, la senal es la reversion: "baja" si subio en las ultimas 5
    ruedas (en dolares) y "alza" si bajo."""
    modelo = _modelo(peso_reversion=1.0)
    info = modelo.fit(_histories())
    assert info["escala_modelo"] > 0 and info["escala_reversion"] > 0
    proveedor = _proveedor(RUEDAS[-1])[0]
    for t, r in modelo.predict_panel(_histories()).items():
        usd = _histories()[t]["close"] / proveedor.reindex(RUEDAS, method="ffill")
        pasado = np.log(usd.iloc[-1] / usd.iloc[-6])
        esperado = "baja" if pasado > 0 else "alza"
        assert r["signal"] == esperado, t
    with pytest.raises(ValueError, match="peso_reversion"):
        TransformerV2Config(peso_reversion=1.5)


def test_perdida_bce_predice_retornos_acotados_por_la_vol() -> None:
    modelo = _modelo(perdida="bce", macro=True, horizonte=10)
    info = modelo.fit(_histories())
    assert info["val_direccion"] >= 0
    for r in modelo.predict_panel(_histories()).values():
        # 2p - 1 por el desvio a 10 ruedas: |retorno| < vol diaria * sqrt(10) (~6%).
        assert abs(np.log(r["predicted_close"] / r["last_close"])) < 0.1
        assert r["horizon_days"] == 10


def test_corrige_un_split_sin_ajustar() -> None:
    """Con un split de 10 a 1 sin ajustar, las predicciones limpiando quedan mucho
    mas cerca de las de la historia correcta que sin limpiar (no identicas: la
    limpieza deja en 0 el retorno real del dia del split)."""
    modelo = _modelo()
    modelo.fit(_histories())
    con_split = _histories()
    df = con_split["GGAL"].copy()
    df.loc[df.index < RUEDAS[-50], ["open", "high", "low", "close"]] *= 10
    con_split["GGAL"] = df

    def retornos(resultados: dict) -> np.ndarray:
        r = [resultados[t] for t in TICKERS]
        return np.log([x["predicted_close"] / x["last_close"] for x in r])

    correcto = retornos(modelo.predict_panel(_histories()))
    limpiando = retornos(modelo.predict_panel(con_split))
    modelo.config = dataclasses.replace(modelo.config, limpiar_saltos=False)
    sin_limpiar = retornos(modelo.predict_panel(con_split))
    error_limpiando = np.abs(limpiando - correcto).mean()
    assert error_limpiando < 0.2 * np.abs(sin_limpiar - correcto).mean()


def test_casos_vacios_y_errores() -> None:
    modelo = _modelo()
    with pytest.raises(NotEnoughDataError, match="entrenado"):
        modelo.predict_panel(_histories())
    pocas = {t: df.iloc[:80] for t, df in _histories().items()}
    with pytest.raises(NotEnoughDataError, match="pocos dias"):
        modelo.fit(pocas)
    modelo.fit({**_histories(), "VACIO": _ohlcv(0).iloc[:0]})
    assert modelo.predict_panel({}) == {}
    with pytest.raises(ApiMlError, match="predict_panel"):
        modelo.predict_df(_ohlcv(0))


def test_fit_es_reproducible_con_la_misma_semilla() -> None:
    corte = 500
    modelo = _modelo(seed=3)
    modelo.fit({t: df.iloc[:corte] for t, df in _histories().items()})
    cortas = modelo.predict_panel({t: df.iloc[:corte] for t, df in _histories().items()})
    otro = _modelo(seed=3)
    otro.fit({t: df.iloc[:corte] for t, df in _histories().items()})
    assert otro.predict_panel({t: df.iloc[:corte] for t, df in _histories().items()}) == cortas


def test_aprende_la_direccion() -> None:
    """Con retornos que revierten la semana anterior, acierta si cada accion sube o baja."""
    histories, close, tickers = _historias_con_reversion()
    n, corte = len(RUEDAS), 450
    config = CHICO | dict(
        ventana=1, epocas=40, learning_rate=3e-3, weight_decay=1e-2, dropout=0.1,
        moneda="ars", seed=1,
    )  # fmt: skip
    modelo = TransformerV2Model(TransformerV2Config(**config))
    info = modelo.fit({tk: df.iloc[:corte] for tk, df in histories.items()})
    assert info["val_direccion"] > 0.55
    pred, real = [], []
    for t in range(corte, n - 5, 5):
        dia = modelo.predict_panel({tk: df.iloc[: t + 1] for tk, df in histories.items()})
        for j, tk in enumerate(tickers):
            pred.append(np.log(dia[tk]["predicted_close"] / dia[tk]["last_close"]))
            real.append(np.log(close[t + 5, j] / close[t, j]))
    assert acierto_balanceado(np.array(pred), np.array(real)) > 0.55


def test_save_y_load(tmp_path) -> None:
    modelo = _modelo()
    modelo.fit(_histories())
    ruta = tmp_path / "v2.pt"
    modelo.save(ruta)
    cargado = TransformerV2Model.load(ruta, proveedor=_proveedor)
    assert isinstance(cargado.config, TransformerV2Config) and cargado.config == modelo.config
    original, nuevo = modelo.predict_panel(_histories()), cargado.predict_panel(_histories())
    for t in original:
        assert nuevo[t]["predicted_close"] == pytest.approx(original[t]["predicted_close"])
        assert nuevo[t]["signal"] == original[t]["signal"]
    # Los artefactos de un modelo no se cargan con el otro.
    assert torch.load(ruta, weights_only=False)["architecture"] == "transformer_v2"
    with pytest.raises(ApiMlError, match="no es un artefacto"):
        PanelTransformerModel.load(ruta)
