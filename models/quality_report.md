# Calidad de los modelos: backtest vs paper trading

Generado el 2026-11-06. Periodo de paper trading analizado: 2026-07-22 a 2026-10-30 (14278 predicciones ya comprobadas, 3761 pendientes). Backtest del 2026-09-03T03:19:34Z.

Lectura rapida: la **accuracy direccional** es el % de veces que el modelo acerto si el precio subia o bajaba (50% = azar). El backtest puede estar inflado porque los modelos ya vieron parte de esa historia al entrenar; el paper trading solo cuenta predicciones hechas antes de conocer el resultado, asi que es la medida real. Con menos de 30 predicciones no se saca ninguna conclusion.

## 1. Brecha backtest vs paper trading

| Modelo (paper) | N paper | Acc. backtest | Acc. paper | Brecha | IC 95% paper | Paper vs azar | Ranking backtest → paper |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| lstm | 2604 | 50.9% | 50.2% | +0.7 pp | 48.3% – 52.1% | indistinguible del azar | 2 → 2 |
| lstm-modal | 576 | 50.9% | 48.1% | +2.8 pp | 44.0% – 52.2% | indistinguible del azar | — |
| xgboost | 2426 | 75.9% | 52.4% | +23.5 pp ⚠️ | 50.4% – 54.4% | mejor que el azar | 1 → 1 |
| xgboost-modal | 441 | 75.9% | 48.1% | +27.9 pp ⚠️ | 43.5% – 52.7% | indistinguible del azar | — |
| transformer | 2532 | 50.0% | 50.2% | -0.2 pp | 48.2% – 52.1% | indistinguible del azar | 3 → 3 |

**Alertas:**

- xgboost: el backtest sobreestima la accuracy en 23.5 pp (umbral 10 pp)
- xgboost-modal: el backtest sobreestima la accuracy en 27.9 pp (umbral 10 pp)

Mismos tickers que el backtest (GGAL, YPFD), para comparar manzanas con manzanas:

| Modelo (paper) | N | Acc. backtest | Acc. paper | Hit rate señales paper | Retorno estrategia paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| lstm | 107 | 50.9% | 53.3% | 73.3% | 2.5% |
| lstm-modal | 22 | 50.9% | 59.1% | 55.6% | 1.9% |
| xgboost | 103 | 75.9% | 44.7% | 38.9% | -4.4% |
| xgboost-modal | 18 | 75.9% | 50.0% | 60.0% | 0.8% |
| transformer | 99 | 50.0% | 56.6% | — | 0.0% |

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
| 2026-W42 | 157 | 40.8% | 14.3% | 95.5% | 50.6% |
| 2026-W43 | 388 | 46.9% | 20.8% | 93.8% | 50.0% |
| 2026-W44 | 174 | 57.5% | 31.6% | 89.1% | 50.5% |

### consensus-aggressive

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W40 | 1 (pocos datos) | 100.0% | — | 100.0% | 100.0% |
| 2026-W41 | 100 | 45.0% | 45.2% | 58.0% | 45.5% |
| 2026-W42 | 163 | 44.8% | 41.2% | 30.1% | 45.1% |
| 2026-W43 | 397 | 51.1% | 40.2% | 56.2% | 48.7% |
| 2026-W44 | 182 | 52.8% | 43.8% | 73.6% | 49.6% |

### consensus-conservative

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W40 | 1 (pocos datos) | 100.0% | — | 100.0% | 100.0% |
| 2026-W41 | 100 | 45.0% | — | 100.0% | 45.5% |
| 2026-W42 | 163 | 44.8% | — | 100.0% | 45.1% |
| 2026-W43 | 397 | 51.1% | — | 100.0% | 48.7% |
| 2026-W44 | 182 | 52.8% | — | 100.0% | 49.6% |

### consensus-moderate

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W40 | 1 (pocos datos) | 100.0% | — | 100.0% | 100.0% |
| 2026-W41 | 100 | 45.0% | 25.0% | 92.0% | 45.5% |
| 2026-W42 | 163 | 44.8% | 39.5% | 73.6% | 45.1% |
| 2026-W43 | 397 | 51.1% | 30.9% | 82.9% | 48.7% |
| 2026-W44 | 182 | 52.8% | 60.0% | 97.2% | 49.6% |

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
| 2026-W42 | 157 | 57.3% | 68.4% | 87.9% | 47.5% |
| 2026-W43 | 388 | 58.5% | 52.0% | 80.7% | 49.3% |
| 2026-W44 | 174 | 63.8% | 73.1% | 46.6% | 50.2% |

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
| 2026-W42 | 90 | 41.1% | 39.4% | 21.1% | 49.3% |
| 2026-W43 | 79 | 51.9% | 55.1% | 12.7% | 49.8% |
| 2026-W44 | 124 | 41.9% | 40.4% | 12.1% | 48.1% |

### macro

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W41 | 64 | 26.6% | 81.0% | 67.2% | 26.6% |

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
| 2026-W42 | 157 | 56.0% | 75.0% | 97.5% | 51.2% |
| 2026-W43 | 328 | 44.8% | 33.3% | 89.9% | 50.3% |
| 2026-W44 | 162 | 48.1% | 22.2% | 88.9% | 50.2% |

### transformer-v1

| Semana | N | Acc. semana | Hit rate señales | % neutral | Acc. acumulada |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-W42 | 1 (pocos datos) | 0.0% | — | 100.0% | 0.0% |
| 2026-W43 | 326 | 47.2% | 32.1% | 91.4% | 47.1% |
| 2026-W44 | 174 | 52.9% | 18.8% | 90.8% | 49.1% |

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
| 2026-W42 | 157 | 63.7% | 75.9% | 81.5% | 49.4% |
| 2026-W43 | 388 | 68.3% | 83.7% | 88.9% | 52.4% |

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
| 2026-W42 | 90 | 41.1% | 41.4% | 35.6% | 48.4% |
| 2026-W43 | 67 | 46.3% | 46.7% | 32.8% | 48.1% |

## 3. Que se probo y si mejoro (por version del modelo)

Solo se listan versiones con al menos 30 predicciones comprobadas. "Mejoro"/"empeoro" significa que los intervalos de confianza no se solapan; si se solapan, la diferencia puede ser ruido.

### arima

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| arima-1-1-1 | 2026-07-22 | 2605 | 50.5% | 48.6% – 52.4% | primera |

### consensus-aggressive

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| consensus-v1 | 2026-10-02 | 843 | 49.6% | 46.2% – 52.9% | primera |

### consensus-conservative

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| consensus-v1 | 2026-10-02 | 843 | 49.6% | 46.2% – 52.9% | primera |

### consensus-moderate

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| consensus-v1 | 2026-10-02 | 843 | 49.6% | 46.2% – 52.9% | primera |

### lstm

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| lstm-20260716T233804Z | 2026-07-22 | 751 | 44.7% | 41.2% – 48.3% | primera |
| lstm-20260831T233222Z | 2026-09-01 | 1353 | 49.5% | 46.8% – 52.1% | sin diferencia significativa |
| lstm-20261020T225237Z | 2026-10-20 | 326 | 58.9% | 53.5% – 64.1% | mejoro |
| lstm-20261026T233034Z | 2026-10-26 | 174 | 63.8% | 56.4% – 70.6% | sin diferencia significativa |

### lstm-modal

Ninguna version alcanza la muestra minima todavia.

### macro

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| macro-20261006T034300Z | 2026-10-07 | 64 | 26.6% | 17.3% – 38.5% | primera |

### transformer

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| transformer-20260716T233905Z | 2026-07-22 | 751 | 46.9% | 43.3% – 50.4% | primera |
| transformer-20260828T040100Z | 2026-09-01 | 1134 | 53.3% | 50.4% – 56.2% | sin diferencia significativa |
| transformer-20261013T014735Z | 2026-10-13 | 219 | 51.6% | 45.0% – 58.1% | sin diferencia significativa |
| transformer-v2-usd | 2026-10-20 | 32 | 37.5% | 22.9% – 54.8% | sin diferencia significativa |
| respaldo:transformer-20261013T014735Z | 2026-10-20 | 396 | 47.5% | 42.6% – 52.4% | sin diferencia significativa |

### transformer-v1

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| transformer-20261013T014735Z | 2026-10-16 | 501 | 49.1% | 44.8% – 53.5% | primera |

### xgboost

| Version | Desde | N | Acc. | IC 95% | vs version anterior |
| --- | --- | ---: | ---: | --- | --- |
| xgb-20260716T233824Z | 2026-07-22 | 747 | 48.1% | 44.5% – 51.6% | primera |
| xgb-20260829T024049Z | 2026-09-01 | 1679 | 54.4% | 52.0% – 56.8% | mejoro |

### xgboost-modal

Ninguna version alcanza la muestra minima todavia.

