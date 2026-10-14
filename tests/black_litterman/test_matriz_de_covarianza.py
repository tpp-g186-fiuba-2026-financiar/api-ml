from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import fake_ohlcv

from src.black_litterman.matriz_de_covarianza import MatrizDeCovarianza


class TestConstruccionBasica:
    def test_shape_correcta(self, historiales_3_tickers):
        tickers = list(historiales_3_tickers.keys())
        m = MatrizDeCovarianza(tickers, historiales_3_tickers)
        assert m.matriz.shape == (3, 3)

    def test_matriz_simetrica(self, historiales_3_tickers):
        tickers = list(historiales_3_tickers.keys())
        m = MatrizDeCovarianza(tickers, historiales_3_tickers)
        assert m.matriz == pytest.approx(m.matriz.T)

    def test_diagonal_positiva(self, historiales_3_tickers):
        tickers = list(historiales_3_tickers.keys())
        m = MatrizDeCovarianza(tickers, historiales_3_tickers)
        assert np.all(np.diag(m.matriz) > 0)

    def test_guarda_tickers_en_el_mismo_orden(self, historiales_3_tickers):
        tickers = list(historiales_3_tickers.keys())
        m = MatrizDeCovarianza(tickers, historiales_3_tickers)
        assert m.tickers == tickers


class TestEstimacion:
    def test_escala_por_el_horizonte(self, historiales_3_tickers):
        tickers = list(historiales_3_tickers.keys())
        diaria = MatrizDeCovarianza(tickers, historiales_3_tickers).matriz
        semanal = MatrizDeCovarianza(tickers, historiales_3_tickers, horizonte=5).matriz
        assert semanal == pytest.approx(5 * diaria)

    def test_shrinkage_mejora_el_condicionamiento(self):
        # Pocas ruedas y activos casi colineales: la muestral queda casi
        # singular, Ledoit-Wolf la vuelve invertible sin problemas.
        base = fake_ohlcv(seed=1, n=30)
        historiales = {f"T{i}": fake_ohlcv(seed=1, n=30) for i in range(3)}
        historiales["T3"] = base.assign(close=base["close"] * 1.0001)
        tickers = list(historiales)
        m = MatrizDeCovarianza(tickers, historiales)

        retornos = np.column_stack(
            [
                np.diff(historiales[t]["close"]) / historiales[t]["close"].to_numpy()[:-1]
                for t in tickers
            ]
        )
        muestral = np.cov(retornos, rowvar=False)
        assert np.linalg.cond(m.matriz) < np.linalg.cond(muestral)

    def test_un_solo_dia_en_comun_lanza_error(self):
        historiales = {
            "A": fake_ohlcv(seed=1, n=10, start="2024-01-01"),
            "B": fake_ohlcv(seed=2, n=10, start="2024-01-09"),
        }
        with pytest.raises(ValueError, match="no hay fechas en comun"):
            MatrizDeCovarianza(["A", "B"], historiales)


class TestAlineacionDeFechas:
    def test_sin_fechas_en_comun_lanza_error(self):
        historiales = {
            "A": fake_ohlcv(seed=1, n=30, start="2024-01-01"),
            "B": fake_ohlcv(seed=2, n=30, start="2030-01-01"),  # rango totalmente disjunto
        }
        with pytest.raises(ValueError, match="no hay fechas en comun"):
            MatrizDeCovarianza(["A", "B"], historiales)

    def test_usa_solo_interseccion_no_rellena(self):
        # B tiene 10 dias menos al principio -- la interseccion debe ser
        # mas chica que el historial completo de A, y el resultado no debe
        # explotar ni inventar datos.
        historiales = {
            "A": fake_ohlcv(seed=1, n=50, start="2024-01-01"),
            "B": fake_ohlcv(seed=2, n=40, start="2024-01-11"),
        }
        m = MatrizDeCovarianza(["A", "B"], historiales)
        assert m.matriz.shape == (2, 2)
        assert not np.isnan(m.matriz).any()

    def test_normaliza_horas_distintas_entre_tickers(self):
        # Reproduce el caso real: BYMA a las 14:00 vs. un commodity a las 04:00.
        # Sin normalizar a solo fecha, esto daba 0 fechas en comun.
        historiales = {
            "GGAL": fake_ohlcv(seed=1, n=40, hour=14),
            "GOLD": fake_ohlcv(seed=2, n=40, hour=4),
        }
        m = MatrizDeCovarianza(["GGAL", "GOLD"], historiales)
        assert m.matriz.shape == (2, 2)
        assert not np.isnan(m.matriz).any()


class TestMatrizGarch:
    def test_reemplaza_diagonal_mantiene_correlacion(self, historiales_3_tickers):
        tickers = list(historiales_3_tickers.keys())
        m = MatrizDeCovarianza(tickers, historiales_3_tickers)

        varianzas_originales = np.diag(m.matriz).copy()
        desvios_originales = np.sqrt(varianzas_originales)
        correlacion_original = m.matriz / np.outer(desvios_originales, desvios_originales)

        class FakeGarch:
            def forecast_variance_horizonte(self, df, horizonte):
                # devuelve una varianza bien distinta a la muestral, para
                # poder chequear que efectivamente se reemplaza.
                return 0.01

        resultado = m.matriz_garch(historiales_3_tickers, FakeGarch())

        # La diagonal ahora tiene que ser la varianza GARCH (0.01), no la muestral.
        assert np.diag(resultado) == pytest.approx(np.full(3, 0.01))

        # La correlacion implicita en el resultado debe seguir siendo la muestral.
        desvios_nuevos = np.sqrt(np.diag(resultado))
        correlacion_nueva = resultado / np.outer(desvios_nuevos, desvios_nuevos)
        assert correlacion_nueva == pytest.approx(correlacion_original)

    def test_llama_forecast_con_el_historial_de_cada_ticker(self, historiales_3_tickers):
        tickers = list(historiales_3_tickers.keys())
        m = MatrizDeCovarianza(tickers, historiales_3_tickers)

        llamados = []

        class FakeGarch:
            def forecast_variance_horizonte(self, df, horizonte):
                llamados.append((df, horizonte))
                return 0.005

        m.matriz_garch(historiales_3_tickers, FakeGarch())
        assert len(llamados) == 3
        for df, horizonte in llamados:
            assert isinstance(df, pd.DataFrame)
            assert horizonte == 1

    def test_pide_el_forecast_al_horizonte_de_la_matriz(self, historiales_3_tickers):
        tickers = list(historiales_3_tickers.keys())
        m = MatrizDeCovarianza(tickers, historiales_3_tickers, horizonte=5)
        horizontes = []

        class FakeGarch:
            def forecast_variance_horizonte(self, df, horizonte):
                horizontes.append(horizonte)
                return 0.005

        m.matriz_garch(historiales_3_tickers, FakeGarch())
        assert horizontes == [5, 5, 5]
