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

    mu, sigma y delta tienen que estar en el mismo horizonte (ej. 5 ruedas)
    para que el termino de retorno y el de riesgo sean comparables.

    peso_maximo limita la concentracion por activo. None (default) = sin
    techo, solo el limite implicito de 1.0 por sum(w) == 1.
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

    # Con retornos y varianzas a pocas ruedas el objetivo vale ~1e-4, por
    # debajo de la tolerancia por defecto de SLSQP (ftol=1e-6): el solver
    # cortaba en la primera iteracion y devolvia w0 (pesos iguales) sin
    # optimizar nada. Se divide el objetivo por la varianza media para
    # llevarlo a orden 1 (no cambia el optimo) y se pide una tolerancia
    # mas fina.
    escala = max(float(np.mean(np.diag(sigma))), 1e-12)

    resultado = minimize(
        lambda w: objetivo(w) / escala,
        w0,
        jac=lambda w: objetivo_grad(w) / escala,
        method="SLSQP",
        bounds=limites,
        constraints=restricciones,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    if not resultado.success:
        raise ValueError(f"el optimizador no convergio: {resultado.message}")

    return resultado.x
