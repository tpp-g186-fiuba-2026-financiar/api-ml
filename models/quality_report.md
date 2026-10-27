# Calidad de los modelos: backtest vs paper trading

Generado el 2026-10-27. Periodo de paper trading analizado: 2026-07-22 a 2026-10-20 (10777 predicciones ya comprobadas, 3858 pendientes). Backtest del 2026-09-03T03:19:34Z.

Lectura rapida: la **accuracy direccional** es el % de veces que el modelo acerto si el precio subia o bajaba (50% = azar). El backtest puede estar inflado porque los modelos ya vieron parte de esa historia al entrenar; el paper trading solo cuenta predicciones hechas antes de conocer el resultado, asi que es la medida real. Con menos de 30 predicciones no se saca ninguna conclusion.

## 1. Brecha backtest vs paper trading

| Modelo (paper) | N paper | Acc. backtest | Acc. paper | Brecha | IC 95% paper | Paper vs azar | Ranking backtest → paper |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| lstm | 2165 | 50.9% | 47.6% | +3.3 pp | 45.5% – 49.7% | peor que el azar | 2 → 3 |
| lstm-modal | 439 | 50.9% | 50.1% | +0.8 pp | 45.5% – 54.8% | indistinguible del azar | — |
| xgboost | 2161 | 75.9% | 49.8% | +26.2 pp ⚠️ | 47.6% – 51.8% | indistinguible del azar | 1 → 2 |
| xgboost-modal | 440 | 75.9% | 47.9% | +28.0 pp ⚠️ | 43.3% – 52.6% | indistinguible del azar | — |
| transformer | 2165 | 50.0% | 50.7% | -0.7 pp | 48.6% – 52.8% | indistinguible del azar | 3 → 1 |

**Alertas:**

- xgboost: el backtest sobreestima la accuracy en 26.2 pp (umbral 10 pp)
- xgboost-modal: el backtest sobreestima la accuracy en 28.0 pp (umbral 10 pp)

Mismos tickers que el backtest (GGAL, YPFD), para comparar manzanas con manzanas:

| Modelo (paper) | N | Acc. backtest | Acc. paper | Hit rate señales paper | Retorno estrategia paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| lstm | 97 | 50.9% | 49.5% | 63.6% | 2.4% |
| lstm-modal | 18 | 50.9% | 50.0% | 42.9% | -0.0% |
| xgboost | 97 | 75.9% | 42.3% | 35.3% | -4.7% |
| xgboost-modal | 18 | 75.9% | 50.0% | 60.0% | 0.8% |
| transformer | 97 | 50.0% | 57.7% | — | 0.0% |

## 2. Evolucion semanal en paper trading

### arima

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W30 | 42 | 50.0% | — | 100.0% | 50.0% |
| 2026-W31 | 110 | 54.5% | — | 100.0% | 53.3% |
| 2026-W32 | 108 | 59.3% | 100.0% | 96.3% | 55.8% |
| 2026-W33 | 142 | 57.0% | 100.0% | 98.6% | 56.2% |
| 2026-W34 | 233 | 50.2% | 47.4% | 91.8% | 54.0% |
| 2026-W35 | 117 | 41.9% | 33.3% | 97.4% | 52.1% |
| 2026-W36 | 121 | 45.5% | 40.0% | 95.9% | 51.2% |
| 2026-W37 | 329 | 49.2% | 57.9% | 94.2% | 50.7% |
| 2026-W38 | 390 | 52.0% | 52.9% | 95.6% | 51.0% |
| 2026-W39 | 133 | 62.4% | 60.0% | 96.2% | 51.9% |
| 2026-W40 | 62 | 48.4% | 85.7% | 88.7% | 51.8% |
| 2026-W41 | 99 | 45.5% | 100.0% | 99.0% | 51.4% |
| 2026-W42 | 156 | 40.4% | 14.3% | 95.5% | 50.6% |
| 2026-W43 | 124 | 46.8% | 14.3% | 94.3% | 50.4% |

### consensus-aggressive

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W40 | 1 (pocos datos) | 100.0% | — | 100.0% | 100.0% |
| 2026-W41 | 100 | 45.0% | 45.2% | 58.0% | 45.5% |
| 2026-W42 | 162 | 44.4% | 41.2% | 29.6% | 44.9% |
| 2026-W43 | 130 | 51.5% | 50.0% | 23.1% | 47.1% |

### consensus-conservative

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W40 | 1 (pocos datos) | 100.0% | — | 100.0% | 100.0% |
| 2026-W41 | 100 | 45.0% | — | 100.0% | 45.5% |
| 2026-W42 | 162 | 44.4% | — | 100.0% | 44.9% |
| 2026-W43 | 130 | 51.5% | — | 100.0% | 47.1% |

### consensus-moderate

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W40 | 1 (pocos datos) | 100.0% | — | 100.0% | 100.0% |
| 2026-W41 | 100 | 45.0% | 25.0% | 92.0% | 45.5% |
| 2026-W42 | 162 | 44.4% | 39.5% | 73.5% | 44.9% |
| 2026-W43 | 130 | 51.5% | 40.5% | 71.5% | 47.1% |

### lstm

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W30 | 42 | 52.4% | 72.7% | 73.8% | 52.4% |
| 2026-W31 | 110 | 52.7% | 46.9% | 70.9% | 52.6% |
| 2026-W32 | 108 | 39.8% | 54.5% | 89.8% | 47.3% |
| 2026-W33 | 142 | 47.9% | 50.0% | 73.2% | 47.5% |
| 2026-W34 | 232 | 33.6% | 20.5% | 66.4% | 42.4% |
| 2026-W35 | 117 | 57.3% | 60.3% | 41.9% | 44.7% |
| 2026-W36 | 121 | 38.0% | 66.7% | 95.0% | 43.8% |
| 2026-W37 | 329 | 52.3% | 31.8% | 93.3% | 46.1% |
| 2026-W38 | 390 | 53.6% | 34.8% | 94.1% | 48.0% |
| 2026-W39 | 133 | 35.3% | 50.0% | 98.5% | 47.0% |
| 2026-W40 | 62 | 38.7% | 33.3% | 95.2% | 46.7% |
| 2026-W41 | 99 | 46.5% | 40.0% | 95.0% | 46.7% |
| 2026-W42 | 156 | 57.0% | 68.4% | 87.8% | 47.5% |
| 2026-W43 | 124 | 50.0% | 45.6% | 62.9% | 47.6% |

### lstm-modal

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W35 | 3 (pocos datos) | 66.7% | 50.0% | 33.3% | 66.7% |
| 2026-W36 | 94 | 56.4% | 52.8% | 23.4% | 56.7% |
| 2026-W37 | 55 | 50.9% | 59.5% | 32.7% | 54.6% |
| 2026-W38 | 1 (pocos datos) | 100.0% | — | 100.0% | 54.9% |
| 2026-W39 | 53 | 35.9% | 30.8% | 26.4% | 50.0% |
| 2026-W40 | 41 | 53.7% | 55.9% | 17.1% | 50.6% |
| 2026-W41 | 36 | 61.1% | 61.3% | 13.9% | 51.9% |
| 2026-W42 | 89 | 40.5% | 39.4% | 20.2% | 49.2% |
| 2026-W43 | 67 | 55.2% | 57.6% | 11.9% | 50.1% |

### transformer

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W30 | 42 | 54.8% | 44.4% | 78.6% | 54.8% |
| 2026-W31 | 110 | 48.2% | 47.1% | 69.1% | 50.0% |
| 2026-W32 | 108 | 28.7% | 26.3% | 64.8% | 41.1% |
| 2026-W33 | 142 | 34.5% | 35.0% | 43.7% | 38.8% |
| 2026-W34 | 232 | 65.1% | 67.6% | 53.4% | 48.4% |
| 2026-W35 | 117 | 38.5% | 23.4% | 59.8% | 46.9% |
| 2026-W36 | 121 | 38.8% | 60.0% | 95.9% | 45.8% |
| 2026-W37 | 329 | 49.9% | 35.3% | 94.8% | 46.9% |
| 2026-W38 | 390 | 67.4% | 53.1% | 91.8% | 51.9% |
| 2026-W39 | 133 | 35.3% | 100.0% | 99.2% | 50.6% |
| 2026-W40 | 62 | 58.1% | — | 100.0% | 50.9% |
| 2026-W41 | 99 | 48.5% | — | 100.0% | 50.8% |
| 2026-W42 | 156 | 56.4% | 75.0% | 97.4% | 51.2% |
| 2026-W43 | 124 | 42.7% | 25.0% | 90.3% | 50.7% |

### transformer-v1

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W43 | 62 | 41.9% | 25.0% | 93.5% | 41.9% |

### xgboost

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W30 | 42 | 33.3% | 20.0% | 76.2% | 33.3% |
| 2026-W31 | 110 | 46.4% | 35.0% | 63.6% | 42.8% |
| 2026-W32 | 108 | 46.3% | 40.8% | 54.6% | 44.2% |
| 2026-W33 | 142 | 38.0% | 37.8% | 31.0% | 42.0% |
| 2026-W34 | 228 | 58.8% | 64.3% | 49.6% | 48.1% |
| 2026-W35 | 117 | 47.9% | 54.7% | 54.7% | 48.1% |
| 2026-W36 | 121 | 38.8% | 36.8% | 84.3% | 46.8% |
| 2026-W37 | 329 | 49.5% | 43.4% | 83.9% | 47.5% |
| 2026-W38 | 390 | 54.1% | 59.7% | 85.4% | 49.1% |
| 2026-W39 | 133 | 33.8% | 6.7% | 88.7% | 48.0% |
| 2026-W40 | 62 | 61.3% | 57.8% | 27.4% | 48.4% |
| 2026-W41 | 99 | 44.4% | 48.3% | 70.7% | 48.2% |
| 2026-W42 | 156 | 63.5% | 75.9% | 81.4% | 49.4% |
| 2026-W43 | 124 | 55.6% | 80.0% | 87.9% | 49.8% |

### xgboost-modal

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W35 | 3 (pocos datos) | 66.7% | 50.0% | 33.3% | 66.7% |
| 2026-W36 | 95 | 51.6% | 54.5% | 30.5% | 52.0% |
| 2026-W37 | 55 | 45.5% | 45.8% | 56.4% | 49.7% |
| 2026-W38 | 1 (pocos datos) | 100.0% | — | 100.0% | 50.0% |
| 2026-W39 | 53 | 45.3% | 41.2% | 35.9% | 48.8% |
| 2026-W40 | 41 | 58.5% | 53.6% | 31.7% | 50.4% |
| 2026-W41 | 36 | 52.8% | 50.0% | 50.0% | 50.7% |
| 2026-W42 | 89 | 40.5% | 41.4% | 34.8% | 48.3% |
| 2026-W43 | 67 | 46.3% | 46.7% | 32.8% | 47.9% |

## 3. Que se probo y si mejoro (por version del modelo)

Solo se listan versiones con al menos 30 predicciones comprobadas. "Mejoro"/"empeoro" significa que los intervalos de confianza no se solapan; si se solapan, la diferencia puede ser ruido.

### arima

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| arima-1-1-1 | 2026-07-22 | 2166 | 50.4% | 48.3% – 52.5% | primera |

### consensus-aggressive

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| consensus-v1 | 2026-10-02 | 393 | 47.1% | 42.2% – 52.0% | primera |

### consensus-conservative

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| consensus-v1 | 2026-10-02 | 393 | 47.1% | 42.2% – 52.0% | primera |

### consensus-moderate

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| consensus-v1 | 2026-10-02 | 393 | 47.1% | 42.2% – 52.0% | primera |

### lstm

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| lstm-20260716T233804Z | 2026-07-22 | 751 | 44.7% | 41.2% – 48.3% | primera |
| lstm-20260831T233222Z | 2026-09-01 | 1352 | 49.4% | 46.8% – 52.1% | sin diferencia significativa |
| lstm-20261020T225237Z | 2026-10-20 | 62 | 43.5% | 31.9% – 55.9% | sin diferencia significativa |

### lstm-modal

Ninguna version alcanza la muestra minima todavia.

### transformer

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| transformer-20260716T233905Z | 2026-07-22 | 751 | 46.9% | 43.3% – 50.4% | primera |
| transformer-20260828T040100Z | 2026-09-01 | 1134 | 53.3% | 50.4% – 56.2% | sin diferencia significativa |
| transformer-20261013T014735Z | 2026-10-13 | 218 | 51.8% | 45.2% – 58.4% | sin diferencia significativa |
| respaldo:transformer-20261013T014735Z | 2026-10-20 | 42 | 42.9% | 29.1% – 57.8% | sin diferencia significativa |

### transformer-v1

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| transformer-20261013T014735Z | 2026-10-20 | 62 | 41.9% | 30.5% – 54.3% | primera |

### xgboost

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| xgb-20260716T233824Z | 2026-07-22 | 747 | 48.1% | 44.5% – 51.6% | primera |
| xgb-20260829T024049Z | 2026-09-01 | 1414 | 50.6% | 48.0% – 53.2% | sin diferencia significativa |

### xgboost-modal

Ninguna version alcanza la muestra minima todavia.

