"""Cartera simulada: efectivo + tenencias, rebalanceo a pesos objetivo con comision."""

from __future__ import annotations

from dataclasses import dataclass, field

# Comision por operacion (compra o venta) sobre el monto operado: aproximacion
# de la comision + derechos de mercado de un broker argentino.
COMISION_DEFAULT = 0.005
CAPITAL_INICIAL_DEFAULT = 1_000_000.0
# No se operan diferencias menores a esto (evita comisiones por redondeos).
MONTO_MINIMO_OPERACION = 1.0


@dataclass
class Cartera:
    nombre: str
    efectivo: float = CAPITAL_INICIAL_DEFAULT
    comision: float = COMISION_DEFAULT
    tenencias: dict[str, float] = field(default_factory=dict)  # ticker -> cantidad
    comisiones_pagadas: float = 0.0
    monto_operado: float = 0.0

    def valor(self, precios: dict[str, float]) -> float:
        return self.efectivo + sum(
            cantidad * precios[ticker] for ticker, cantidad in self.tenencias.items()
        )

    def pesos(self, precios: dict[str, float]) -> dict[str, float]:
        total = self.valor(precios)
        if total <= 0:
            return {}
        return {t: q * precios[t] / total for t, q in self.tenencias.items() if q > 0}

    def rebalancear(self, objetivo: dict[str, float], precios: dict[str, float]) -> list[dict]:
        """Lleva la cartera a ``objetivo`` (ticker -> peso, suma <= 1; el resto en efectivo).

        Primero vende y despues compra, cobrando ``comision`` sobre cada monto
        operado. Las compras se escalan para que el efectivo alcance a pagar
        las comisiones: nunca queda efectivo negativo ni se usa apalancamiento.
        Fracciones de accion permitidas. Devuelve las operaciones hechas.
        """
        suma = sum(objetivo.values())
        if suma > 1 + 1e-9 or any(w < -1e-12 for w in objetivo.values()):
            raise ValueError(f"pesos objetivo invalidos (suma={suma})")
        total = self.valor(precios)
        operaciones: list[dict] = []

        actuales = {t: q * precios[t] for t, q in self.tenencias.items()}
        deseados = {t: w * total for t, w in objetivo.items() if w > 0}

        for ticker in sorted(set(actuales) | set(deseados)):
            diferencia = deseados.get(ticker, 0.0) - actuales.get(ticker, 0.0)
            if diferencia < -MONTO_MINIMO_OPERACION:
                operaciones.append(self._vender(ticker, -diferencia, precios[ticker]))

        compras = {
            t: deseados[t] - actuales.get(t, 0.0)
            for t in deseados
            if deseados[t] - actuales.get(t, 0.0) > MONTO_MINIMO_OPERACION
        }
        necesario = sum(compras.values()) * (1 + self.comision)
        escala = min(1.0, self.efectivo / necesario) if necesario > 0 else 1.0
        for ticker in sorted(compras):
            operaciones.append(self._comprar(ticker, compras[ticker] * escala, precios[ticker]))

        self.tenencias = {t: q for t, q in self.tenencias.items() if q > 1e-12}
        if -1e-6 < self.efectivo < 0:  # redondeo de punto flotante al gastar todo
            self.efectivo = 0.0
        return operaciones

    def _vender(self, ticker: str, monto: float, precio: float) -> dict:
        cantidad = min(monto / precio, self.tenencias.get(ticker, 0.0))
        monto = cantidad * precio
        costo = monto * self.comision
        self.tenencias[ticker] = self.tenencias.get(ticker, 0.0) - cantidad
        self.efectivo += monto - costo
        self.comisiones_pagadas += costo
        self.monto_operado += monto
        return {"ticker": ticker, "lado": "venta", "cantidad": cantidad, "precio": precio,
                "monto": monto, "comision": costo}  # fmt: skip

    def _comprar(self, ticker: str, monto: float, precio: float) -> dict:
        costo = monto * self.comision
        self.tenencias[ticker] = self.tenencias.get(ticker, 0.0) + monto / precio
        self.efectivo -= monto + costo
        self.comisiones_pagadas += costo
        self.monto_operado += monto
        return {"ticker": ticker, "lado": "compra", "cantidad": monto / precio, "precio": precio,
                "monto": monto, "comision": costo}  # fmt: skip

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "efectivo": self.efectivo,
            "comision": self.comision,
            "tenencias": self.tenencias,
            "comisiones_pagadas": self.comisiones_pagadas,
            "monto_operado": self.monto_operado,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Cartera:
        return cls(**data)
