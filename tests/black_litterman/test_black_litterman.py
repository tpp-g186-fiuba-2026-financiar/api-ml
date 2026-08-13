from __future__ import annotations

import numpy as np
import pytest
from conftest import make_predict_function

from src.black_litterman.black_litterman import (
    MIN_RUEDAS,
    BlackLittermanPrediction,
    Prediccion,
    construir_omega_tradicional,
    entry,
    get_data,
    get_predicciones,
    obtener_precios_actuales,
)
from src.black_litterman.cartera_ancla import CarteraAncla, TipoCarteraAncla
from src.black_litterman.matriz_de_covarianza import MatrizDeCovarianza
from src.schemas import PerfilRiesgo, TrendResponse, Usuario


# --------------------------------------------------------------------------- #
# Prediccion / get_predicciones
# --------------------------------------------------------------------------- #
class TestPrediccion:
    def test_calcula_retorno_no_precio_crudo(self):
        # Regresion: en una version anterior esto guardaba predicted_close
        # crudo en vez del retorno -- Q quedaba en la escala equivocada.
        t = TrendResponse(
            symbol="GGAL",
            signal="alza",
            horizon_days=5,
            last_close=100.0,
            predicted_close=105.0,
            condition="neutral",
            as_of="2025-01-01",
            model_version="test",
        )
        p = Prediccion(t)
        assert p.prediccion == pytest.approx(0.05)

    def test_retorno_negativo(self):
        t = TrendResponse(
            symbol="GGAL",
            signal="baja",
            horizon_days=5,
            last_close=100.0,
            predicted_close=95.0,
            condition="neutral",
            as_of="2025-01-01",
            model_version="test",
        )
        p = Prediccion(t)
        assert p.prediccion == pytest.approx(-0.05)

    def test_guarda_ticker(self):
        t = TrendResponse(
            symbol="YPFD",
            signal="neutral",
            horizon_days=5,
            last_close=400.0,
            predicted_close=400.0,
            condition="neutral",
            as_of="2025-01-01",
            model_version="test",
        )
        p = Prediccion(t)
        assert p.ticker == "YPFD"


class TestGetPredicciones:
    def test_arma_q_en_el_mismo_orden_que_tickers(self):
        tickers = ["GGAL", "YPFD", "PAMP"]
        predict = make_predict_function({"GGAL": 0.05, "YPFD": -0.01, "PAMP": 0.0})
        q = get_predicciones(tickers, predict)
        assert q == pytest.approx([0.05, -0.01, 0.0])

    def test_devuelve_lista_no_objetos(self):
        tickers = ["GGAL"]
        predict = make_predict_function({"GGAL": 0.02})
        q = get_predicciones(tickers, predict)
        assert all(isinstance(x, float) for x in q)


# --------------------------------------------------------------------------- #
# obtener_precios_actuales
# --------------------------------------------------------------------------- #
class TestObtenerPreciosActuales:
    def test_toma_el_close_mas_reciente(self, historiales_3_tickers):
        precios = obtener_precios_actuales(historiales_3_tickers)
        for ticker, df in historiales_3_tickers.items():
            assert precios[ticker] == pytest.approx(float(df.sort_index()["close"].iloc[-1]))

    def test_dataframe_vacio_lanza_error(self):
        import pandas as pd

        historiales = {"GGAL": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])}
        with pytest.raises(ValueError, match="no tiene ningun dato"):
            obtener_precios_actuales(historiales)

    def test_no_asume_orden_ya_ordenado(self):
        # Si el DataFrame viniera desordenado, tiene que ordenar antes de
        # tomar el ultimo -- no confiar en el orden de entrada.
        import pandas as pd

        idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [1, 2, 3],
                "low": [1, 2, 3],
                "close": [30, 10, 20],
                "volume": [1, 1, 1],
            },
            index=idx,
        )
        precios = obtener_precios_actuales({"X": df})
        assert precios["X"] == 30.0  # corresponde a 2024-01-03, la fecha mas reciente


# --------------------------------------------------------------------------- #
# construir_omega_tradicional
# --------------------------------------------------------------------------- #
class TestConstruirOmegaTradicional:
    def test_omega_es_diagonal(self):
        sigma = np.array([[4e-4, 1e-5, 0.0], [1e-5, 5e-4, 0.0], [0.0, 0.0, 3e-4]])
        omega = construir_omega_tradicional(sigma, tau=0.05)
        off_diagonal = omega - np.diag(np.diag(omega))
        assert np.all(off_diagonal == 0)

    def test_omega_ii_es_tau_por_sigma_ii(self):
        sigma = np.diag([4e-4, 5e-4, 3e-4])
        tau = 0.05
        omega = construir_omega_tradicional(sigma, tau)
        assert np.diag(omega) == pytest.approx(tau * np.diag(sigma))

    def test_mayor_tau_da_mayor_incertidumbre(self):
        sigma = np.diag([4e-4, 5e-4])
        omega_bajo = construir_omega_tradicional(sigma, tau=0.01)
        omega_alto = construir_omega_tradicional(sigma, tau=0.1)
        assert np.all(np.diag(omega_alto) > np.diag(omega_bajo))


# --------------------------------------------------------------------------- #
# get_data: filtro por MIN_RUEDAS + inyeccion de fetch
# --------------------------------------------------------------------------- #
def _fake_historial(rows: int, base: float = 100.0):
    """DataFrame OHLCV minimo con `rows` ruedas, para forzar el caso de
    historial insuficiente (ej. ECOG con 17 ruedas) sin depender de
    parametros de fake_ohlcv que no sabemos si soporta largo variable.
    """
    import pandas as pd

    idx = pd.date_range("2025-01-01", periods=rows, freq="D")
    closes = [base + i for i in range(rows)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * rows,
        },
        index=idx,
    )


class TestGetDataFiltraPorMinimoDeRuedas:
    def test_excluye_ticker_con_pocas_ruedas(self, historiales_3_tickers):
        historiales_completos = {
            **historiales_3_tickers,
            "ECOG": _fake_historial(rows=17),
        }

        def fetch_hist(ticker, days=None):
            return historiales_completos[ticker]

        tickers = [*historiales_3_tickers.keys(), "ECOG"]
        tickers_validos, historiales, precios = get_data(tickers, fetch_hist)

        assert "ECOG" not in tickers_validos
        assert "ECOG" not in historiales
        assert "ECOG" not in precios
        assert set(tickers_validos) == set(historiales_3_tickers.keys())

    def test_no_filtra_acciones_con_historial_suficiente(self, historiales_3_tickers):
        def fetch_hist(ticker, days=None):
            return historiales_3_tickers[ticker]

        tickers = list(historiales_3_tickers.keys())
        tickers_validos, _, _ = get_data(tickers, fetch_hist)

        assert set(tickers_validos) == set(historiales_3_tickers.keys())

    def test_respeta_min_ruedas_pasado_por_parametro(self, historiales_3_tickers):
        # Con un min_ruedas mas laxo que el default, un historial que antes
        # se excluia ahora deberia entrar.
        historiales_completos = {
            **historiales_3_tickers,
            "ECOG": _fake_historial(rows=17),
        }

        def fetch_hist(ticker, days=None):
            return historiales_completos[ticker]

        tickers = [*historiales_3_tickers.keys(), "ECOG"]
        tickers_validos, _, _ = get_data(tickers, fetch_hist, min_ruedas=10)

        assert "ECOG" in tickers_validos

    def test_todos_los_tickers_insuficientes_lanza_error(self):
        def fetch_hist(ticker, days=None):
            return _fake_historial(rows=5)

        with pytest.raises(ValueError, match="minimo de ruedas"):
            get_data(["GGAL", "YPFD"], fetch_hist)

    def test_constante_min_ruedas_coincide_con_garch(self):
        # Test de "documentacion viva": MIN_RUEDAS tiene que ir en linea con
        # MIN_ROWS de garch_volatility.py -- si se desincronizan, un ticker
        # puede pasar el filtro de get_data y de todos modos explotarle a
        # GARCH mas abajo (o al reves, excluirse de mas).
        from src.garch_volatility import MIN_ROWS

        assert MIN_RUEDAS == MIN_ROWS


# --------------------------------------------------------------------------- #
# BlackLittermanPrediction: internals
# --------------------------------------------------------------------------- #
@pytest.fixture
def bl_prediction(usuario_con_cartera, historiales_3_tickers, precios_3_tickers):
    tickers = list(historiales_3_tickers.keys())
    cartera_ancla = CarteraAncla(usuario_con_cartera, tickers, precios_3_tickers)
    matriz = MatrizDeCovarianza(tickers, historiales_3_tickers)
    q = [0.02, -0.01, 0.0]
    omega = construir_omega_tradicional(matriz.matriz, tau=0.05)
    return BlackLittermanPrediction(usuario_con_cartera, matriz, cartera_ancla, q, omega)


class TestObtenerAlfa:
    @pytest.mark.parametrize(
        "perfil,esperado",
        [
            (PerfilRiesgo.CONSERVADOR, 0.75),
            (PerfilRiesgo.MODERADO, 0.5),
            (PerfilRiesgo.ARRIESGADO, 0.25),
        ],
    )
    def test_mapeo_de_perfil(self, perfil, esperado, historiales_3_tickers, precios_3_tickers):
        usuario = Usuario(perfil_riesgo=perfil, tenencias=[{"ticker": "GGAL", "cantidad": 1}])
        tickers = list(historiales_3_tickers.keys())
        cartera_ancla = CarteraAncla(usuario, tickers, precios_3_tickers)
        matriz = MatrizDeCovarianza(tickers, historiales_3_tickers)
        omega = construir_omega_tradicional(matriz.matriz, tau=0.05)
        bl = BlackLittermanPrediction(usuario, matriz, cartera_ancla, [0.0, 0.0, 0.0], omega)
        assert bl.perfil_de_riesgo == esperado

    def test_orden_correcto_conservador_mayor_que_arriesgado(self):
        # Regresion conceptual: conservador tiene que pesar MAS (se pega mas
        # al equilibrio), no menos, que arriesgado.
        from src.black_litterman.black_litterman import (
            ALFA_ARRIESGADO,
            ALFA_CONSERVADOR,
            ALFA_MODERADO,
        )

        assert ALFA_CONSERVADOR > ALFA_MODERADO > ALFA_ARRIESGADO


class TestInvTauSigmaCache:
    def test_segundo_llamado_no_explota(self, bl_prediction):
        # Regresion del bug "== None" contra un ndarray ya calculado
        # (ValueError: the truth value of an array...).
        primero = bl_prediction._inv_tau_sigma()
        segundo = bl_prediction._inv_tau_sigma()
        assert primero is segundo  # mismo objeto cacheado, no recalculado

    def test_valor_correcto(self, bl_prediction):
        esperado = np.linalg.inv(bl_prediction.tau * bl_prediction.matriz_de_covarianza.matriz)
        assert bl_prediction._inv_tau_sigma() == pytest.approx(esperado)

    def test_bl_mu_no_explota_al_llamar_dos_veces_internamente(self, bl_prediction):
        # bl_mu() llama a _inv_tau_sigma() mas de una vez -- si el cacheo
        # estuviera roto, esto es justamente lo que explotaba antes.
        mu = bl_prediction.bl_mu()
        assert mu.shape == (3,)
        assert not np.isnan(mu).any()


class TestGetPi:
    def test_pi_es_delta_por_sigma_por_w_ref(self, bl_prediction):
        esperado = bl_prediction.perfil_de_riesgo * np.dot(
            bl_prediction.matriz_de_covarianza.matriz, bl_prediction.cartera_ancla.vector
        )
        assert bl_prediction.pi == pytest.approx(esperado)

    def test_pi_reproduce_w_ref_si_omega_es_infinito(
        self, usuario_con_cartera, historiales_3_tickers, precios_3_tickers
    ):
        # Chequeo de sanidad clasico de Black-Litterman: si las vistas no
        # tienen ninguna confianza (Omega enorme), el resultado optimo
        # deberia reproducir la cartera de referencia.
        tickers = list(historiales_3_tickers.keys())
        cartera_ancla = CarteraAncla(usuario_con_cartera, tickers, precios_3_tickers)
        matriz = MatrizDeCovarianza(tickers, historiales_3_tickers)
        omega_gigante = np.eye(3) * 1e12  # practicamente sin confianza
        bl = BlackLittermanPrediction(
            usuario_con_cartera, matriz, cartera_ancla, [0.5, 0.5, 0.5], omega_gigante
        )
        mu_bl = bl.bl_mu()
        assert mu_bl == pytest.approx(bl.pi, abs=1e-6)


class TestBlSigma:
    def test_shape_y_simetrica(self, bl_prediction):
        sigma_bl = bl_prediction.bl_sigma()
        assert sigma_bl.shape == (3, 3)
        assert sigma_bl == pytest.approx(sigma_bl.T)

    def test_mayor_o_igual_a_sigma_original(self, bl_prediction):
        # Sigma_BL = Sigma + termino de incertidumbre (siempre suma, nunca resta).
        sigma_bl = bl_prediction.bl_sigma()
        diagonal_original = np.diag(bl_prediction.matriz_de_covarianza.matriz)
        diagonal_bl = np.diag(sigma_bl)
        assert np.all(diagonal_bl >= diagonal_original - 1e-12)


class TestPredecir:
    def test_devuelve_pesos_validos(self, bl_prediction):
        w = bl_prediction.predecir()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-8)
        assert np.all(w <= 1.0 + 1e-8)


# --------------------------------------------------------------------------- #
# entry(): integracion de punta a punta
# --------------------------------------------------------------------------- #
class TestEntryEndToEnd:
    def _fetch_hist(self, historiales):
        def fetch_hist(ticker, days=None):
            return historiales[ticker]

        return fetch_hist

    def test_devuelve_tickers_y_pesos_validos(self, usuario_con_cartera, historiales_3_tickers):
        fetch_hist = self._fetch_hist(historiales_3_tickers)
        tickers_in = list(historiales_3_tickers.keys())
        predict = make_predict_function({"GGAL": 0.02, "YPFD": -0.01, "PAMP": 0.0})

        tickers, pesos = entry(
            usuario_con_cartera, predict, tickers=tickers_in, fetch_hist=fetch_hist
        )

        assert set(tickers) == set(historiales_3_tickers.keys())
        assert len(pesos) == len(tickers)
        assert pesos.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(pesos >= -1e-8)
        assert np.all(pesos <= 1.0 + 1e-8)

    def test_excluye_ticker_con_pocas_ruedas_del_resultado_final(
        self, usuario_con_cartera, historiales_3_tickers
    ):
        historiales_completos = {
            **historiales_3_tickers,
            "ECOG": _fake_historial(rows=17),
        }
        fetch_hist = self._fetch_hist(historiales_completos)
        tickers_in = [*historiales_3_tickers.keys(), "ECOG"]
        predict = make_predict_function({})  # todas las vistas neutras (0%)

        tickers, pesos = entry(
            usuario_con_cartera, predict, tickers=tickers_in, fetch_hist=fetch_hist
        )

        assert "ECOG" not in tickers
        assert len(pesos) == len(tickers)

    def test_usuario_sin_cartera_no_explota(self, usuario_sin_cartera, historiales_3_tickers):
        # Con la cartera ancla cayendo a EQUAL_WEIGHT por default, un
        # usuario nuevo (sin tenencias) tiene que poder recibir una
        # recomendacion igual, no un error.
        fetch_hist = self._fetch_hist(historiales_3_tickers)
        tickers_in = list(historiales_3_tickers.keys())
        predict = make_predict_function({"GGAL": 0.03})

        tickers, pesos = entry(
            usuario_sin_cartera, predict, tickers=tickers_in, fetch_hist=fetch_hist
        )
        assert pesos.sum() == pytest.approx(1.0, abs=1e-6)

    def test_tipo_cartera_ancla_explicito_se_respeta(
        self, usuario_con_cartera, historiales_3_tickers
    ):
        fetch_hist = self._fetch_hist(historiales_3_tickers)
        tickers_in = list(historiales_3_tickers.keys())
        predict = make_predict_function({})

        # Fuerzo EQUAL_WEIGHT aunque el usuario tenga cartera propia -- no
        # deberia fallar ni ignorar el parametro.
        tickers, pesos = entry(
            usuario_con_cartera,
            predict,
            tipo_cartera_ancla=TipoCarteraAncla.EQUAL_WEIGHT,
            tickers=tickers_in,
            fetch_hist=fetch_hist,
        )
        assert pesos.sum() == pytest.approx(1.0, abs=1e-6)

    def test_con_garch_model_no_explota_y_cambia_el_resultado(
        self, usuario_con_cartera, historiales_3_tickers
    ):
        fetch_hist = self._fetch_hist(historiales_3_tickers)
        tickers_in = list(historiales_3_tickers.keys())
        predict = make_predict_function({"GGAL": 0.02, "YPFD": 0.019})

        # Variazas MUY distintas entre si (no un unico valor uniforme) para
        # forzar que el activo mas favorecido pueda cambiar realmente --
        # con retornos casi iguales, quien tenga menor varianza GARCH deberia
        # ganar terreno.
        variancias_garch = {"GGAL": 0.05, "YPFD": 1e-6, "PAMP": 0.05}

        class FakeGarch:
            def forecast_daily_variance(self, df):
                # identificamos el ticker por el precio base (unico por ticker
                # en la fixture), ya que forecast_daily_variance solo recibe el df
                ultimo_close = float(df["close"].iloc[0])
                if ultimo_close == pytest.approx(100.0, abs=5):
                    return variancias_garch["GGAL"]
                if ultimo_close == pytest.approx(400.0, abs=20):
                    return variancias_garch["YPFD"]
                return variancias_garch["PAMP"]

        _, pesos_sin_garch = entry(
            usuario_con_cartera, predict, tickers=tickers_in, fetch_hist=fetch_hist
        )
        _, pesos_con_garch = entry(
            usuario_con_cartera,
            predict,
            garch_model=FakeGarch(),
            tickers=tickers_in,
            fetch_hist=fetch_hist,
        )

        assert pesos_con_garch.sum() == pytest.approx(1.0, abs=1e-6)
        assert not np.allclose(pesos_sin_garch, pesos_con_garch)

    def test_garch_model_reemplaza_la_diagonal_efectivamente(
        self, usuario_con_cartera, historiales_3_tickers
    ):
        # Chequeo mas directo/estructural: la matriz que termina usando
        # BlackLittermanPrediction tiene que tener la diagonal GARCH, no la
        # muestral -- sin depender de si eso alcanza para cambiar w*.
        fetch_hist = self._fetch_hist(historiales_3_tickers)
        tickers_in = list(historiales_3_tickers.keys())
        predict = make_predict_function({})

        capturada = {}

        class FakeGarch:
            def forecast_daily_variance(self, df):
                return 0.0123

        import src.black_litterman.black_litterman as bl_mod

        original = bl_mod.BlackLittermanPrediction

        class SpyBlackLitterman(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                capturada["matriz"] = self.matriz_de_covarianza.matriz

        bl_mod.BlackLittermanPrediction = SpyBlackLitterman
        try:
            entry(
                usuario_con_cartera,
                predict,
                garch_model=FakeGarch(),
                tickers=tickers_in,
                fetch_hist=fetch_hist,
            )
        finally:
            bl_mod.BlackLittermanPrediction = original

        assert np.diag(capturada["matriz"]) == pytest.approx(np.full(3, 0.0123))

    def test_perfiles_distintos_pueden_dar_resultados_distintos(self, historiales_3_tickers):
        # No siempre van a diferir (depende de mu/Sigma), pero con una vista
        # fuerte y concentrada, mayor delta (conservador) deberia diversificar
        # al menos tanto como uno mas agresivo.
        fetch_hist = self._fetch_hist(historiales_3_tickers)
        tickers_in = list(historiales_3_tickers.keys())
        predict = make_predict_function({"GGAL": 0.15})

        resultados = {}
        for perfil in [PerfilRiesgo.CONSERVADOR, PerfilRiesgo.MODERADO, PerfilRiesgo.ARRIESGADO]:
            usuario = Usuario(
                perfil_riesgo=perfil,
                tenencias=[{"ticker": "GGAL", "cantidad": 10}, {"ticker": "YPFD", "cantidad": 10}],
            )
            _, pesos = entry(usuario, predict, tickers=tickers_in, fetch_hist=fetch_hist)
            resultados[perfil] = pesos

        concentracion_conservador = np.max(resultados[PerfilRiesgo.CONSERVADOR])
        concentracion_arriesgado = np.max(resultados[PerfilRiesgo.ARRIESGADO])
        assert concentracion_conservador <= concentracion_arriesgado + 1e-6