"""Optimizador de cartera con restricciones (sin ventas en corto, sin
apalancamiento), para reemplazar la formula cerrada sin restricciones.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def optimizar_pesos(mu: np.ndarray, sigma: np.ndarray, delta: float) -> np.ndarray:
    """Resuelve la cartera optima de media-varianza con restricciones:

    - sin ventas en corto:  w_i >= 0 para todo i
    - sin apalancamiento:   sum(w) == 1

    Maximiza mu.w - (delta/2) * w'.Sigma.w (equivalente a minimizar su
    negativo, que es lo que hace scipy.optimize.minimize).
    """
    n = len(mu)

    def objetivo(w):
        return -(mu @ w - (delta / 2) * w @ sigma @ w)

    def objetivo_grad(w):
        return -(mu - delta * sigma @ w)

    restricciones = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},  # sin apalancamiento
    ]
    limites = [(0.0, 1.0) for _ in range(n)]  # sin cortos (piso 0), ademas techo 100%

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