import pytest

from src.simulacion.cartera import Cartera

PRECIOS = {"A": 10.0, "B": 20.0}


def test_rebalancear_desde_efectivo_cobra_comision_y_no_deja_efectivo_negativo() -> None:
    cartera = Cartera("x", efectivo=1000.0, comision=0.01)
    operaciones = cartera.rebalancear({"A": 0.5, "B": 0.5}, PRECIOS)

    assert [op["lado"] for op in operaciones] == ["compra", "compra"]
    assert cartera.efectivo == pytest.approx(0.0, abs=1e-9)
    assert cartera.efectivo >= 0
    # Lo pagado en comisiones sale del valor de la cartera.
    assert cartera.valor(PRECIOS) == pytest.approx(1000.0 - cartera.comisiones_pagadas)
    pesos = cartera.pesos(PRECIOS)
    assert pesos["A"] == pytest.approx(pesos["B"])


def test_rebalancear_a_vacio_vende_todo() -> None:
    cartera = Cartera("x", efectivo=1000.0, comision=0.0)
    cartera.rebalancear({"A": 1.0}, PRECIOS)
    operaciones = cartera.rebalancear({}, PRECIOS)

    assert [op["lado"] for op in operaciones] == ["venta"]
    assert cartera.tenencias == {}
    assert cartera.efectivo == pytest.approx(1000.0)


def test_rebalancear_parcial_deja_el_resto_en_efectivo() -> None:
    cartera = Cartera("x", efectivo=1000.0, comision=0.0)
    cartera.rebalancear({"A": 0.25}, PRECIOS)
    assert cartera.efectivo == pytest.approx(750.0)
    assert cartera.monto_operado == pytest.approx(250.0)


@pytest.mark.parametrize("objetivo", [{"A": 0.7, "B": 0.7}, {"A": -0.1}])
def test_rebalancear_rechaza_pesos_invalidos(objetivo) -> None:
    with pytest.raises(ValueError):
        Cartera("x").rebalancear(objetivo, PRECIOS)


def test_pesos_de_cartera_sin_valor() -> None:
    assert Cartera("x", efectivo=0.0).pesos(PRECIOS) == {}


def test_to_dict_from_dict() -> None:
    cartera = Cartera("x", efectivo=1000.0)
    cartera.rebalancear({"A": 0.5}, PRECIOS)
    assert Cartera.from_dict(cartera.to_dict()) == cartera
