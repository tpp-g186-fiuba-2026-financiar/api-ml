"""Clasificador SVM de direccion (sube/baja al dia siguiente), por ticker,
ajustado al vuelo en cada request.

Adaptado del SVM que vive en el repo `models` (Modal). A diferencia de
LSTM/XGBoost/ARIMA, este modelo NO predice un retorno a horizonte de varios
dias: es un clasificador binario que predice si el cierre de MANANA va a
ser mayor o menor al de HOY, a partir de dos features del dia (Open-Close,
High-Low). No tiene una magnitud de retorno asociada -- no se fuerza al
formato de `TrendResponse` (que exige expected_return/predicted_close),
tiene su propio endpoint y forma de respuesta.

Convencion de signo: replica tal cual la logica ya validada por el equipo
en `models/modelos/svm_model.py` (con su fix "invert signal"): clase 0 =>
sube (alza), clase 1 => baja.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import svm

from src.errors import NotEnoughDataError

MIN_ROWS = 60


@dataclass
class SvmDirectionModel:
    @property
    def version(self) -> str:
        return "svm-rbf"

    def predict_df(self, df: pd.DataFrame) -> dict:
        if len(df) < MIN_ROWS:
            raise NotEnoughDataError(f"se necesitan al menos {MIN_ROWS} ruedas, hay {len(df)}")

        features = pd.DataFrame(
            {
                "open_close": df["close"] - df["open"],
                "high_low": df["high"] - df["low"],
            }
        ).reset_index(drop=True)
        # target[i] = 1 si el cierre del dia siguiente es mayor al de hoy.
        target = np.where(df["close"].shift(-1) > df["close"], 1, 0)[:-1]

        # La ultima fila no tiene "manana" conocido: se usa solo para predecir.
        X_train = features.iloc[:-1]
        x_last = features.iloc[[-1]]

        clf = svm.SVC()
        clf.fit(X_train, target)
        predicted_class = int(clf.predict(x_last)[0])
        decision = float(clf.decision_function(x_last)[0])

        signal = "alza" if predicted_class == 0 else "baja"
        confidence = float(min(1.0, abs(decision)))

        return {
            "signal": signal,
            "horizon_days": 1,
            "confidence": round(confidence, 4),
            "last_close": round(float(df["close"].iloc[-1]), 4),
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
        }
