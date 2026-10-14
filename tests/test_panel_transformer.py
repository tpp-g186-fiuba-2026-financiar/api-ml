import numpy as np
import pandas as pd
import pytest
import torch

from src.errors import ApiMlError, NotEnoughDataError
from src.panel_gbm import features_y_contexto
from src.panel_transformer import (
    PanelTransformerConfig,
    PanelTransformerModel,
    _RedPanel,
    _ventanas,
    ic_pearson_por_dia,
)

RUEDAS = pd.bdate_range("2023-01-02", periods=600)
TICKERS = ["GGAL", "BMA", "YPFD", "PAMP", "ALUA", "TXAR", "CEPU", "EDN"]
CHICO = dict(ventana=10, d_model=16, nhead=2, dim_feedforward=32, epocas=3, batch_dias=32)


def _ohlcv(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(RUEDAS))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": rng.integers(500, 2000, len(RUEDAS)).astype(float)},
        index=RUEDAS,
    )  # fmt: skip


def _histories() -> dict[str, pd.DataFrame]:
    return {t: _ohlcv(i) for i, t in enumerate(TICKERS)}


def _macro() -> pd.DataFrame:
    dias = pd.date_range(RUEDAS[0] - pd.Timedelta(days=400), RUEDAS[-1], freq="D")
    return pd.DataFrame(
        {"ccl": np.linspace(300, 1500, len(dias)), "oficial": np.linspace(200, 1400, len(dias)),
         "riesgo_pais": np.linspace(2000, 600, len(dias)), "badlar": 40.0},
        index=dias,
    )  # fmt: skip


def _proveedor(hasta):
    macro = _macro()
    macro = macro[macro.index <= hasta]
    return macro["ccl"], macro


def _modelo(**extra) -> PanelTransformerModel:
    return PanelTransformerModel(PanelTransformerConfig(**(CHICO | extra)), proveedor=_proveedor)


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"perdida": "mse"}, "perdida"),
        ({"d_model": 30, "nhead": 4}, "divisible"),
        ({"ventana": 100}, "ventana"),
        ({"universo": "todos"}, "universo"),
        ({"n_semillas": 0}, "n_semillas"),
        ({"capas_temporales": 0}, "capa temporal"),
    ],
)
def test_config_invalida(extra, match) -> None:
    with pytest.raises(ValueError, match=match):
        PanelTransformerConfig(**extra)


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"macro": True, "emb_ticker": True, "perdida": "huber", "capas_cruzadas": 2},
        {"capas_cruzadas": 0},
        {"moneda": "ars", "perdida": "ic", "n_semillas": 2, "vida_media_anios": None,
         "emb_sector": False, "epocas": 8, "paciencia": 1},
    ],
)  # fmt: skip
def test_fit_y_predict_panel(extra) -> None:
    modelo = _modelo(**extra)
    info = modelo.fit(_histories())
    assert info["dias_train"] > 300 and info["dias_val"] > 50
    assert len(modelo._redes) == modelo.config.n_semillas == len(info["epoca_mejor"])
    assert (modelo._normalizador_macro is not None) == modelo.config.macro
    dfs = _histories()
    dfs["NOESTA"] = _ohlcv(99)  # fuera del universo: se ignora
    resultados = modelo.predict_panel(dfs)
    assert set(resultados) == set(TICKERS)
    for t, r in resultados.items():
        assert r["last_close"] == pytest.approx(dfs[t]["close"].iloc[-1], rel=1e-3)
        assert np.isfinite(r["predicted_close"])
    assert (
        modelo.resumen()["version"] == modelo.version == f"panel-transformer-{modelo.config.moneda}"
    )


def test_ticker_sin_dato_el_ultimo_dia_queda_afuera() -> None:
    modelo = _modelo()
    modelo.fit(_histories())
    dfs = _histories()
    dfs["GGAL"] = dfs["GGAL"].iloc[:-1]
    resultados = modelo.predict_panel(dfs)
    assert "GGAL" not in resultados and len(resultados) == 7


def test_casos_vacios_y_errores() -> None:
    modelo = _modelo()
    with pytest.raises(NotEnoughDataError, match="entrenado"):
        modelo.predict_panel(_histories())
    with pytest.raises(ApiMlError, match="predict_panel"):
        modelo.predict_df(_ohlcv(0))
    with pytest.raises(ApiMlError, match="proveedor"):
        PanelTransformerModel(PanelTransformerConfig(**CHICO)).fit(_histories())
    pocas = {t: df.iloc[:80] for t, df in _histories().items()}
    with pytest.raises(NotEnoughDataError, match="pocos dias"):
        modelo.fit(pocas)
    modelo.fit(_histories())
    assert modelo.predict_panel({}) == {}
    assert modelo.predict_panel({"NOESTA": _ohlcv(0)}) == {}
    quietos = {t: df.assign(volume=0.0) for t, df in _histories().items()}
    assert modelo.predict_panel(quietos) == {}
    cortos = {t: df.iloc[:9] for t, df in _histories().items()}  # sin ventana completa
    assert modelo.predict_panel(cortos) == {}


def test_ic_pearson_por_dia_ignora_celdas_fuera_de_la_mascara() -> None:
    rng = np.random.default_rng(0)
    score = torch.tensor(rng.normal(size=(3, 6)), requires_grad=True)
    y = torch.tensor(rng.normal(size=(3, 6)))
    m = torch.ones(3, 6, dtype=torch.bool)
    m[1, :2] = False
    ic = ic_pearson_por_dia(score, y, m)
    s, t = score.detach().numpy(), y.numpy()
    assert ic[0].item() == pytest.approx(np.corrcoef(s[0], t[0])[0, 1], abs=1e-6)
    assert ic[1].item() == pytest.approx(np.corrcoef(s[1, 2:], t[1, 2:])[0, 1], abs=1e-6)
    ic.sum().backward()
    assert score.grad is not None and torch.all(score.grad[1, :2] == 0)


def test_ventanas_miran_hacia_atras() -> None:
    x = torch.arange(10.0)[:, None, None].expand(10, 2, 1)
    v = _ventanas(x, torch.tensor([4, 9]), ventana=3)
    assert v.shape == (2, 2, 3, 1)
    assert v[0, 0, :, 0].tolist() == [2.0, 3.0, 4.0] and v[1, 1, :, 0].tolist() == [7.0, 8.0, 9.0]


@pytest.mark.parametrize("macro", [False, True])
def test_ticker_ausente_no_cambia_a_los_demas(macro) -> None:
    """La mascara de la atencion entre acciones aisla a los tickers ausentes."""
    cfg = PanelTransformerConfig(**CHICO, emb_ticker=True)
    torch.manual_seed(0)
    red = _RedPanel(cfg, n_features=4, n_macro=3 if macro else 0, n_tickers=5).eval()
    x = torch.randn(2, 5, cfg.ventana, 4)
    presentes = torch.ones(2, 5, dtype=torch.bool)
    presentes[:, 2] = False
    args = (torch.arange(5), torch.tensor([0, 1, 1, 2, 3]), torch.randn(2, 3) if macro else None)
    with torch.no_grad():
        antes = red(x, presentes, *args)
        x[:, 2] = 100.0
        despues = red(x, presentes, *args)
    otros = [0, 1, 3, 4]
    assert torch.allclose(antes[:, otros], despues[:, otros], atol=1e-6)


def test_anti_leakage_el_score_de_t_no_ve_datos_posteriores() -> None:
    """Con el modelo ya entrenado, cambiar precios posteriores a ``t`` no cambia
    el score de ``t`` (features, ventana y token macro solo miran hacia atras)."""
    modelo = _modelo(macro=True)
    modelo.fit({t: df.iloc[:450] for t, df in _histories().items()})
    t = 500
    alterados = {}
    for i, (tk, df) in enumerate(_histories().items()):
        df = df.copy()
        factor = np.exp(np.random.default_rng(i).normal(0, 0.05, len(df) - t - 1))
        df.iloc[t + 1 :, :4] = df.iloc[t + 1 :, :4].mul(factor, axis=0)
        alterados[tk] = df

    def score_en_t(histories):
        fp, ctx = features_y_contexto(histories, modelo.config, _proveedor)
        x, tickers, sectores, macro = modelo._tensores(fp, ctx)
        datos = {
            "x": torch.tensor(x),
            "presentes": torch.tensor(fp.mascara),
            "tickers": torch.tensor(tickers),
            "sectores": torch.tensor(sectores),
            "macro": torch.tensor(macro, dtype=torch.float32),
        }
        with torch.no_grad():
            return modelo._forward(modelo._redes[0], datos, torch.tensor([t]))[0].numpy()

    original, alterado = score_en_t(_histories()), score_en_t(alterados)
    assert np.allclose(original, alterado, atol=1e-5) and original.std() > 0


def test_aprende_una_senal_de_reversion() -> None:
    """Con retornos que revierten la semana anterior, el modelo ordena bien."""
    rng = np.random.default_rng(7)
    n, k = len(RUEDAS), 12
    tickers = [*TICKERS, "SUPV", "BBAR", "TGSU2", "LOMA"]
    r = rng.normal(0, 0.02, (n, k))
    for t in range(5, n):
        r[t] -= 0.15 * r[t - 5 : t].sum(axis=0)
    close = 100 * np.exp(np.cumsum(r, axis=0))
    histories = {
        tk: pd.DataFrame({"open": close[:, j], "high": close[:, j] * 1.01,
                          "low": close[:, j] * 0.99, "close": close[:, j], "volume": 1000.0},
                         index=RUEDAS)
        for j, tk in enumerate(tickers)
    }  # fmt: skip
    corte = 450
    modelo = PanelTransformerModel(
        PanelTransformerConfig(**(CHICO | {"epocas": 15, "moneda": "ars", "seed": 1}))
    )
    info = modelo.fit({tk: df.iloc[:corte] for tk, df in histories.items()})
    assert info["val_ic"] > 0.1
    ics = []
    for t in range(corte, n - 5, 5):
        pred = modelo.predict_panel({tk: df.iloc[: t + 1] for tk, df in histories.items()})
        p = pd.Series(
            {tk: np.log(v["predicted_close"] / v["last_close"]) for tk, v in pred.items()}
        )
        real = pd.Series(
            {tk: np.log(close[t + 5, j] / close[t, j]) for j, tk in enumerate(tickers)}
        )
        ics.append(p.rank().corr(real[p.index].rank()))
    assert np.mean(ics) > 0.1
