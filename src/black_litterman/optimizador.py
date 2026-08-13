"""Optimizador de cartera con restricciones (sin ventas en corto, sin
apalancamiento), para reemplazar la formula cerrada sin restricciones.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def optimizar_pesos(
    mu: np.ndarray, sigma: np.ndarray, delta: float, peso_maximo: float | None = None
) -> np.ndarray:
    """Resuelve la cartera optima de media-varianza con restricciones:

    - sin ventas en corto:  w_i >= 0 para todo i
    - sin apalancamiento:   sum(w) == 1
    - (opcional) concentracion maxima: w_i <= peso_maximo para todo i

    Maximiza mu.w - (delta/2) * w'.Sigma.w (equivalente a minimizar su
    negativo, que es lo que hace scipy.optimize.minimize).

    peso_maximo es una salvaguarda independiente de que delta este bien
    calibrado: aunque el objetivo este bien escalado, activos muy
    correlacionados entre si (Sigma mal condicionada) pueden seguir
    empujando a una solucion muy concentrada. None (default) = sin techo,
    solo el limite implicito de 1.0 por sum(w) == 1.
    """
    n = len(mu)

    techo = 1.0 if peso_maximo is None else peso_maximo
    if techo * n < 1.0:
        raise ValueError(
            f"peso_maximo={techo} es incompatible con {n} tickers "
            f"(peso_maximo * n = {techo * n} < 1.0, no se puede sumar 1 sin superar el techo)"
        )

    def objetivo(w):
        return -(mu @ w - (delta / 2) * w @ sigma @ w)

    def objetivo_grad(w):
        return -(mu - delta * sigma @ w)

    restricciones = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},  # sin apalancamiento
    ]
    limites = [(0.0, techo) for _ in range(n)]  # sin cortos (piso 0), techo configurable

    w0 = np.full(n, 1.0 / n)  # punto inicial: pesos iguales

    resultado = minimize(
        objetivo,
        w0,
        jac=objetivo_grad,
        method="SLSQP",
        bounds=limites,
        constraints=restricciones,
    )

    if not resultado.success:
        raise ValueError(f"el optimizador no convergio: {resultado.message}")

    return resultado.x
