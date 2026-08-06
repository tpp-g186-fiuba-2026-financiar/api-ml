from __future__ import annotations

import numpy as np
import pytest

from src.black_litterman.cartera_ancla import CarteraAncla, TipoCarteraAncla
from src.schemas import PerfilRiesgo, Usuario


class TestVectorPropia:
    def test_pesos_proporcionales_al_valor_invertido(self, usuario_con_cartera, precios_3_tickers):
        tickers = ["GGAL", "YPFD", "PAMP"]
        ancla = CarteraAncla(usuario_con_cartera, tickers, precios_3_tickers)

        assert ancla.tipo == TipoCarteraAncla.PROPIA
        valor_ggal = 150 * precios_3_tickers["GGAL"]
        valor_ypfd = 40 * precios_3_tickers["YPFD"]
        total = valor_ggal + valor_ypfd
        assert ancla.vector[0] == pytest.approx(valor_ggal / total)
        assert ancla.vector[1] == pytest.approx(valor_ypfd / total)
        assert ancla.vector[2] == pytest.approx(0.0)

    def test_suma_uno(self, usuario_con_cartera, precios_3_tickers):
        tickers = ["GGAL", "YPFD", "PAMP"]
        ancla = CarteraAncla(usuario_con_cartera, tickers, precios_3_tickers)
        assert ancla.vector.sum() == pytest.approx(1.0)

    def test_cartera_vacia_lanza_value_error(self, usuario_sin_cartera, precios_3_tickers):
        tickers = ["GGAL", "YPFD", "PAMP"]
        with pytest.raises(ValueError, match="valor positivo"):
            CarteraAncla(usuario_sin_cartera, tickers, precios_3_tickers, tipo=TipoCarteraAncla.PROPIA)

    def test_ticker_sin_tenencia_pesa_cero(self, precios_3_tickers):
        usuario = Usuario(
            perfil_riesgo=PerfilRiesgo.MODERADO,
            tenencias=[{"ticker": "GGAL", "cantidad": 10}],
        )
        tickers = ["GGAL", "YPFD", "PAMP"]
        ancla = CarteraAncla(usuario, tickers, precios_3_tickers)
        assert ancla.vector[1] == 0.0
        assert ancla.vector[2] == 0.0


class TestVectorEqualWeight:
    def test_pesos_iguales_y_suman_uno(self, usuario_con_cartera, precios_3_tickers):
        tickers = ["GGAL", "YPFD", "PAMP"]
        ancla = CarteraAncla(
            usuario_con_cartera, tickers, precios_3_tickers, tipo=TipoCarteraAncla.EQUAL_WEIGHT
        )
        assert ancla.vector == pytest.approx(np.full(3, 1 / 3))

    def test_ignora_tenencias_reales(self, usuario_sin_cartera, precios_3_tickers):
        # usuario_sin_cartera no tiene ninguna tenencia, pero EQUAL_WEIGHT
        # no deberia depender de eso ni fallar.
        tickers = ["GGAL", "YPFD", "PAMP"]
        ancla = CarteraAncla(
            usuario_sin_cartera, tickers, precios_3_tickers, tipo=TipoCarteraAncla.EQUAL_WEIGHT
        )
        assert ancla.vector.sum() == pytest.approx(1.0)


class TestVectorMercado:
    def test_no_implementado(self, usuario_con_cartera, precios_3_tickers):
        tickers = ["GGAL", "YPFD", "PAMP"]
        with pytest.raises(NotImplementedError):
            CarteraAncla(
                usuario_con_cartera, tickers, precios_3_tickers, tipo=TipoCarteraAncla.MERCADO
            )


class TestEleccionPorDefecto:
    def test_usa_propia_si_hay_valor_positivo(self, usuario_con_cartera, precios_3_tickers):
        tickers = ["GGAL", "YPFD", "PAMP"]
        ancla = CarteraAncla(usuario_con_cartera, tickers, precios_3_tickers)
        assert ancla.tipo == TipoCarteraAncla.PROPIA

    def test_cae_a_equal_weight_si_no_hay_tenencias(self, usuario_sin_cartera, precios_3_tickers):
        tickers = ["GGAL", "YPFD", "PAMP"]
        ancla = CarteraAncla(usuario_sin_cartera, tickers, precios_3_tickers)
        assert ancla.tipo == TipoCarteraAncla.EQUAL_WEIGHT
        assert ancla.vector == pytest.approx(np.full(3, 1 / 3))

    def test_no_lanza_error_con_tenencias_vacias_por_default(
        self, usuario_sin_cartera, precios_3_tickers
    ):
        # A diferencia de pedir PROPIA explicito, el default nunca deberia
        # explotar por una cartera vacia -- cae a EQUAL_WEIGHT solo.
        tickers = ["GGAL", "YPFD", "PAMP"]
        CarteraAncla(usuario_sin_cartera, tickers, precios_3_tickers)  # no debe lanzar
