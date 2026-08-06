from __future__ import annotations

import numpy as np
import pytest

from src.black_litterman.optimizador import optimizar_pesos


def sigma_diagonal(n: int, var: float = 4e-4) -> np.ndarray:
    return np.eye(n) * var


class TestRestricciones:
    def test_pesos_suman_uno(self):
        mu = np.array([0.01, 0.02, -0.005])
        sigma = sigma_diagonal(3)
        w = optimizar_pesos(mu, sigma, delta=2.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_sin_cortos(self):
        mu = np.array([0.01, -0.05, 0.02, -0.1])
        sigma = sigma_diagonal(4)
        w = optimizar_pesos(mu, sigma, delta=2.0)
        assert np.all(w >= -1e-8)

    def test_sin_apalancamiento(self):
        mu = np.array([0.05, 0.05, 0.05])
        sigma = sigma_diagonal(3)
        w = optimizar_pesos(mu, sigma, delta=2.0)
        assert np.all(w <= 1.0 + 1e-8)


class TestComportamientoEsperado:
    def test_activo_dominante_se_lleva_todo_el_peso(self):
        # Un activo con mucho mejor retorno esperado y misma varianza que
        # el resto deberia concentrar el 100% (dado un delta chico que no
        # penalice tanto el riesgo).
        mu = np.array([0.20, 0.01, 0.01])
        sigma = sigma_diagonal(3)
        w = optimizar_pesos(mu, sigma, delta=1.0)
        assert w[0] == pytest.approx(1.0, abs=1e-4)
        assert w[1] == pytest.approx(0.0, abs=1e-4)
        assert w[2] == pytest.approx(0.0, abs=1e-4)

    def test_retornos_iguales_reparte_por_riesgo(self):
        # Con el mismo mu para todos pero varianzas distintas, deberia
        # favorecer al de menor varianza.
        mu = np.array([0.02, 0.02])
        sigma = np.array([[1e-4, 0.0], [0.0, 9e-4]])  # activo 0 mucho menos volatil
        w = optimizar_pesos(mu, sigma, delta=5.0)
        assert w[0] > w[1]

    def test_mayor_delta_diversifica_mas_con_retornos_similares(self):
        # A mayor aversion al riesgo, con retornos parecidos, deberia
        # diversificar mas en vez de concentrar en el de mayor retorno.
        mu = np.array([0.021, 0.020, 0.019])
        sigma = np.array(
            [[4e-4, 1e-4, 1e-4], [1e-4, 4e-4, 1e-4], [1e-4, 1e-4, 4e-4]]
        )
        w_bajo_delta = optimizar_pesos(mu, sigma, delta=1.0)
        w_alto_delta = optimizar_pesos(mu, sigma, delta=500.0)

        concentracion_bajo = np.max(w_bajo_delta)
        concentracion_alto = np.max(w_alto_delta)
        assert concentracion_alto < concentracion_bajo


class TestCasosBorde:
    def test_un_solo_activo_se_lleva_todo(self):
        mu = np.array([0.03])
        sigma = sigma_diagonal(1)
        w = optimizar_pesos(mu, sigma, delta=2.0)
        assert w[0] == pytest.approx(1.0)

    def test_lanza_error_si_no_converge(self, monkeypatch):
        import src.black_litterman.optimizador as opt_mod

        class FakeResultado:
            success = False
            message = "fallo simulado"

        monkeypatch.setattr(opt_mod, "minimize", lambda *a, **k: FakeResultado())

        with pytest.raises(ValueError, match="no convergio"):
            optimizar_pesos(np.array([0.01, 0.02]), sigma_diagonal(2), delta=2.0)
