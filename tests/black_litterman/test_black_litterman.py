from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import fake_ohlcv, make_predict_function

from src.black_litterman.black_litterman import (
    AVERSION_AL_RIESGO,
    DELTA_MERCADO,
    MIN_RUEDAS,
    PESO_MAXIMO_POR_PERFIL,
    TAU,
    BlackLittermanPrediction,
    Prediccion,
    construir_omega,
    entry,
    get_data,
    get_predicciones,
    horizonte_comun,
    obtener_aversion_al_riesgo,
    obtener_precios_actuales,
    tasa_al_horizonte,
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
        assert p.horizonte == 5


class TestGetPredicciones:
    def test_mantiene_el_orden_de_tickers(self):
        tickers = ["GGAL", "YPFD", "PAMP"]
        predict = make_predict_function({"GGAL": 0.05, "YPFD": -0.01, "PAMP": 0.0})
        predicciones = get_predicciones(tickers, predict)
        assert [p.ticker for p in predicciones] == tickers
        assert [p.prediccion for p in predicciones] == pytest.approx([0.05, -0.01, 0.0])


class TestHorizonteComun:
    def test_devuelve_el_horizonte_compartido(self):
        predicciones = get_predicciones(["GGAL", "YPFD"], make_predict_function({}))
        assert horizonte_comun(predicciones) == 5

    def test_horizontes_distintos_lanza_error(self):
        predicciones = get_predicciones(["GGAL", "YPFD"], make_predict_function({}))
        predicciones[1].horizonte = 10
        with pytest.raises(ValueError, match="horizontes distintos"):
            horizonte_comun(predicciones)


class TestTasaAlHorizonte:
    def test_tna_a_5_ruedas(self):
        assert tasa_al_horizonte(0.252, 5) == pytest.approx(0.005)

    def test_tasa_cero(self):
        assert tasa_al_horizonte(0.0, 5) == 0.0


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
# construir_omega
# --------------------------------------------------------------------------- #
class TestConstruirOmega:
    def test_omega_es_diagonal(self):
        sigma = np.array([[4e-4, 1e-5, 0.0], [1e-5, 5e-4, 0.0], [0.0, 0.0, 3e-4]])
        omega = construir_omega(sigma, tau=0.05, confianza=0.3)
        off_diagonal = omega - np.diag(np.diag(omega))
        assert np.all(off_diagonal == 0)

    def test_confianza_media_es_el_omega_clasico(self):
        # c = 0.5 => Omega_ii = tau * Sigma_ii (mezcla mitad prior, mitad vista).
        sigma = np.diag([4e-4, 5e-4, 3e-4])
        omega = construir_omega(sigma, tau=0.05, confianza=0.5)
        assert np.diag(omega) == pytest.approx(0.05 * np.diag(sigma))

    def test_menos_confianza_da_mas_incertidumbre(self):
        sigma = np.diag([4e-4, 5e-4])
        omega_baja = construir_omega(sigma, tau=0.05, confianza=0.05)
        omega_alta = construir_omega(sigma, tau=0.05, confianza=0.8)
        assert np.all(np.diag(omega_baja) > np.diag(omega_alta))

    def test_acepta_una_confianza_por_ticker(self):
        sigma = np.diag([4e-4, 4e-4])
        omega = construir_omega(sigma, tau=0.05, confianza=np.array([0.5, 0.2]))
        assert omega[0, 0] == pytest.approx(0.05 * 4e-4)
        assert omega[1, 1] == pytest.approx(0.05 * 4e-4 * 4)

    @pytest.mark.parametrize("confianza", [0.0, 1.0, -0.1, 1.5])
    def test_confianza_fuera_de_rango_lanza_error(self, confianza):
        with pytest.raises(ValueError, match="confianza"):
            construir_omega(np.diag([4e-4]), tau=0.05, confianza=confianza)

    def test_mueve_el_posterior_una_fraccion_c_hacia_la_vista(
        self, usuario_sin_cartera, historiales_3_tickers, precios_3_tickers
    ):
        # Con un solo activo (sin correlaciones que mezclen vistas) el
        # retorno posterior es exactamente Pi + c * (Q - Pi).
        historiales = {"GGAL": historiales_3_tickers["GGAL"]}
        matriz = MatrizDeCovarianza(["GGAL"], historiales, horizonte=5)
        ancla = CarteraAncla(usuario_sin_cartera, ["GGAL"], precios_3_tickers)
        c = 0.2
        omega = construir_omega(matriz.matriz, TAU, c)
        bl = BlackLittermanPrediction(usuario_sin_cartera, matriz, ancla, [0.04], omega)
        assert bl.bl_mu()[0] == pytest.approx(bl.pi[0] + c * (0.04 - bl.pi[0]))


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
    matriz = MatrizDeCovarianza(tickers, historiales_3_tickers, horizonte=5)
    q = [0.02, -0.01, 0.0]
    omega = construir_omega(matriz.matriz, TAU, 0.3)
    return BlackLittermanPrediction(usuario_con_cartera, matriz, cartera_ancla, q, omega)


class TestPerfilDeRiesgo:
    @pytest.mark.parametrize("perfil", list(PerfilRiesgo))
    def test_usa_aversion_y_techo_del_perfil(
        self, perfil, historiales_3_tickers, precios_3_tickers
    ):
        usuario = Usuario(perfil_riesgo=perfil, tenencias=[{"ticker": "GGAL", "cantidad": 1}])
        tickers = list(historiales_3_tickers.keys())
        cartera_ancla = CarteraAncla(usuario, tickers, precios_3_tickers)
        matriz = MatrizDeCovarianza(tickers, historiales_3_tickers)
        omega = construir_omega(matriz.matriz, TAU, 0.3)
        bl = BlackLittermanPrediction(usuario, matriz, cartera_ancla, [0.0, 0.0, 0.0], omega)
        assert bl.aversion_al_riesgo == AVERSION_AL_RIESGO[perfil]
        assert bl.peso_maximo == PESO_MAXIMO_POR_PERFIL[perfil]

    def test_conservador_mas_averso_y_con_menos_techo_que_arriesgado(self):
        assert (
            AVERSION_AL_RIESGO[PerfilRiesgo.CONSERVADOR]
            > AVERSION_AL_RIESGO[PerfilRiesgo.MODERADO]
            > AVERSION_AL_RIESGO[PerfilRiesgo.ARRIESGADO]
        )
        assert (
            PESO_MAXIMO_POR_PERFIL[PerfilRiesgo.CONSERVADOR]
            < PESO_MAXIMO_POR_PERFIL[PerfilRiesgo.MODERADO]
            < PESO_MAXIMO_POR_PERFIL[PerfilRiesgo.ARRIESGADO]
        )

    def test_perfil_desconocido_lanza_error(self):
        with pytest.raises(ValueError, match="perfil de riesgo desconocido"):
            obtener_aversion_al_riesgo("inexistente")


class TestGetPi:
    def test_pi_es_delta_mercado_por_sigma_por_w_ancla(self, bl_prediction):
        esperado = DELTA_MERCADO * (
            bl_prediction.matriz_de_covarianza.matriz @ bl_prediction.cartera_ancla.vector
        )
        assert bl_prediction.pi == pytest.approx(esperado)

    def test_pi_no_depende_del_perfil_del_usuario(self, historiales_3_tickers, precios_3_tickers):
        # El equilibrio es del mercado: el perfil solo entra en el optimizador.
        tickers = list(historiales_3_tickers.keys())
        matriz = MatrizDeCovarianza(tickers, historiales_3_tickers)
        omega = construir_omega(matriz.matriz, TAU, 0.3)
        pis = []
        for perfil in PerfilRiesgo:
            usuario = Usuario(perfil_riesgo=perfil, tenencias=[{"ticker": "GGAL", "cantidad": 1}])
            ancla = CarteraAncla(usuario, tickers, precios_3_tickers)
            pis.append(BlackLittermanPrediction(usuario, matriz, ancla, [0.0] * 3, omega).pi)
        assert pis[0] == pytest.approx(pis[1])
        assert pis[0] == pytest.approx(pis[2])


class TestBlMu:
    def test_vistas_iguales_al_prior_devuelven_el_prior(self, bl_prediction):
        # Si las vistas coinciden con el equilibrio no hay nada que mover,
        # cualquiera sea Omega.
        bl_prediction.q = bl_prediction.pi.copy()
        assert bl_prediction.bl_mu() == pytest.approx(bl_prediction.pi)

    def test_sin_confianza_en_las_vistas_queda_en_el_prior(
        self, usuario_con_cartera, historiales_3_tickers, precios_3_tickers
    ):
        tickers = list(historiales_3_tickers.keys())
        cartera_ancla = CarteraAncla(usuario_con_cartera, tickers, precios_3_tickers)
        matriz = MatrizDeCovarianza(tickers, historiales_3_tickers)
        omega_gigante = np.eye(3) * 1e12  # practicamente sin confianza
        bl = BlackLittermanPrediction(
            usuario_con_cartera, matriz, cartera_ancla, [0.5, 0.5, 0.5], omega_gigante
        )
        assert bl.bl_mu() == pytest.approx(bl.pi, abs=1e-6)

    def test_coincide_con_la_formula_cerrada(self, bl_prediction):
        tau_sigma_inv = np.linalg.inv(bl_prediction.tau * bl_prediction.sigma)
        omega_inv = np.linalg.inv(bl_prediction.omega)
        esperado = np.linalg.inv(tau_sigma_inv + omega_inv) @ (
            tau_sigma_inv @ bl_prediction.pi + omega_inv @ bl_prediction.q
        )
        assert bl_prediction.bl_mu() == pytest.approx(esperado)


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
            def forecast_variance_horizonte(self, df, horizonte):
                # identificamos el ticker por el precio base (unico por ticker
                # en la fixture), ya que el forecast solo recibe el df
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
            def forecast_variance_horizonte(self, df, horizonte):
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

    def test_vista_fuerte_gana_peso_sobre_la_ancla(self, usuario_sin_cartera):
        # Regresion: con el delta del optimizador inflado (x1000) las vistas
        # no movian nada y salia la cartera de minima varianza. Con un
        # universo de 6 acciones y ancla equal weight, la que tiene una vista
        # muy positiva tiene que terminar con mas que su 1/6.
        historiales = {f"T{i}": fake_ohlcv(seed=10 + i, n=300) for i in range(6)}
        tickers = list(historiales)
        predict = make_predict_function({"T0": 0.10})

        _, pesos = entry(
            Usuario(perfil_riesgo=PerfilRiesgo.ARRIESGADO, tenencias=[]),
            predict,
            tickers=tickers,
            fetch_hist=self._fetch_hist(historiales),
        )
        assert pesos[0] > 1 / 6 + 0.02

    def test_descuenta_la_tasa_libre_de_riesgo_de_las_vistas(
        self, usuario_con_cartera, historiales_3_tickers, monkeypatch
    ):
        import src.black_litterman.black_litterman as bl_mod

        capturado = {}
        original = bl_mod.BlackLittermanPrediction

        class SpyBlackLitterman(original):
            def __init__(self, usuario, matriz, ancla, q, omega, **kwargs):
                capturado["q"] = list(q)
                super().__init__(usuario, matriz, ancla, q, omega, **kwargs)

        monkeypatch.setattr(bl_mod, "BlackLittermanPrediction", SpyBlackLitterman)
        tickers_in = list(historiales_3_tickers.keys())
        entry(
            usuario_con_cartera,
            make_predict_function({"GGAL": 0.02, "YPFD": -0.01, "PAMP": 0.0}),
            tickers=tickers_in,
            fetch_hist=self._fetch_hist(historiales_3_tickers),
            tasa_libre_riesgo_anual=0.252,  # 0.5% a 5 ruedas
        )
        assert capturado["q"] == pytest.approx([0.015, -0.015, -0.005])

    def test_sigma_queda_al_horizonte_de_las_vistas(
        self, usuario_con_cartera, historiales_3_tickers, monkeypatch
    ):
        import src.black_litterman.black_litterman as bl_mod

        capturado = {}
        original = bl_mod.BlackLittermanPrediction

        class SpyBlackLitterman(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                capturado["matriz"] = self.matriz_de_covarianza

        monkeypatch.setattr(bl_mod, "BlackLittermanPrediction", SpyBlackLitterman)
        tickers_in = list(historiales_3_tickers.keys())
        entry(
            usuario_con_cartera,
            make_predict_function({}),
            tickers=tickers_in,
            fetch_hist=self._fetch_hist(historiales_3_tickers),
        )
        diaria = MatrizDeCovarianza(tickers_in, historiales_3_tickers).matriz
        assert capturado["matriz"].horizonte == 5
        assert capturado["matriz"].matriz == pytest.approx(5 * diaria)


def test_get_data_downloads_in_parallel_but_keeps_ticker_order() -> None:
    import threading
    import time

    from src.black_litterman import black_litterman as bl

    seen = set()

    def slow_fetch(ticker, days):
        seen.add(threading.get_ident())
        time.sleep(0.05)
        idx = pd.bdate_range("2024-01-01", periods=80)
        return pd.DataFrame({"close": np.linspace(100, 110, 80)}, index=idx)

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    started = time.perf_counter()
    validos, historiales, _ = bl.get_data(tickers, slow_fetch)
    elapsed = time.perf_counter() - started

    assert validos == tickers
    assert list(historiales) == tickers
    assert elapsed < 0.05 * len(tickers)  # en serie tardaria >= 0.2 s
    assert len(seen) > 1


def test_get_predicciones_keeps_ticker_order_when_run_in_parallel() -> None:
    from src.black_litterman import black_litterman as bl

    class FakeTrend:
        def __init__(self, symbol, ratio):
            self.symbol = symbol
            self.last_close = 100.0
            self.predicted_close = 100.0 * ratio
            self.horizon_days = 5

    ratios = {"AAA": 1.01, "BBB": 0.98, "CCC": 1.05}
    q = bl.get_predicciones(list(ratios), lambda t: FakeTrend(t, ratios[t]))
    assert [p.prediccion for p in q] == pytest.approx([0.01, -0.02, 0.05])
