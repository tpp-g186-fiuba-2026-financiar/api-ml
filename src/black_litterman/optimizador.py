"""Optimizador de cartera con restricciones (sin ventas en corto, sin
apalancamiento), para reemplazar la formula cerrada sin restricciones.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# Suavizado de |x| para que el costo de operar sea derivable cerca de 0.
_EPS_COSTO = 1e-8


def optimizar_pesos(
    mu: np.ndarray,
    sigma: np.ndarray,
    delta: float,
    peso_maximo: float | None = None,
    *,
    pesos_actuales: np.ndarray | None = None,
    costo: float = 0.0,
    permitidos: np.ndarray | None = None,
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

    ``costo`` (fraccion por unidad operada, ej. 0.006) descuenta del objetivo
    ``costo * sum|w - pesos_actuales|``: solo conviene mover plata si la
    mejora esperada paga la comision. ``permitidos`` (mascara booleana) deja
    en 0 los tickers que no pueden entrar a la cartera.
    """
    n = len(mu)
    actuales = np.zeros(n) if pesos_actuales is None else np.asarray(pesos_actuales)
    mascara = np.ones(n, dtype=bool) if permitidos is None else np.asarray(permitidos, bool)

    techo = 1.0 if peso_maximo is None else peso_maximo
    m = int(mascara.sum())
    if techo * m < 1.0:
        raise ValueError(
            f"peso_maximo={techo} es incompatible con {m} tickers "
            f"(peso_maximo * n = {techo * m} < 1.0, no se puede sumar 1 sin superar el techo)"
        )

    def objetivo(w):
        d = w - actuales
        friccion = costo * np.sqrt(d * d + _EPS_COSTO).sum()
        return -(mu @ w - (delta / 2) * w @ sigma @ w) + friccion

    def objetivo_grad(w):
        d = w - actuales
        return -(mu - delta * sigma @ w) + costo * d / np.sqrt(d * d + _EPS_COSTO)

    restricciones = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},  # sin apalancamiento
    ]
    # sin cortos (piso 0), techo configurable; los no permitidos quedan en 0
    limites = [(0.0, techo if mascara[i] else 0.0) for i in range(n)]

    w0 = np.where(mascara, 1.0 / mascara.sum(), 0.0)  # punto inicial: pesos iguales

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
