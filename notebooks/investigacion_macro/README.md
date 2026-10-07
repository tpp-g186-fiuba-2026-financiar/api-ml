# Investigación: ¿se puede predecir si una acción sube o baja?

Resumen de lo que se midió (octubre 2026) para decidir qué modelos predicen mejor y por qué
existe el modelo `macro`. Todo es **fuera de muestra** (walk-forward: el modelo se entrena solo con
datos anteriores a lo que predice), con intervalos de confianza por bloques de mes.

## Punto de partida

Los modelos existentes (LSTM, XGBoost, Transformer, ARIMA) acertaban 44-52% en paper trading: como
tirar una moneda. Un "modelo" que dijera "sube" todos los días acertaba 55-61% (según el horizonte),
porque en pesos las acciones suben más días de los que bajan. La vara a superar no es el 50%.

## Qué NO funcionó (probado)

| Intento | Resultado |
| --- | --- |
| Indicadores técnicos (RSI, medias, MACD, Bollinger, volumen, gaps, rachas…) | AUC 0.50-0.55 a cualquier horizonte |
| Un modelo por acción vs uno para todas | igual (AUC 0.525 vs 0.526) |
| Horizontes de 1, 2, 3, 5, 10 ruedas | sin señal de baja (precisión 40-46% vs base 46-47%) |
| Regresión logística con las variables macro | AUC 0.52-0.56: hacen falta interacciones (árboles) |
| LSTM / GRU con ventanas de 20 días y las variables macro | AUC 0.46-0.51 |
| Sumar mercado global (S&P 500, VIX, ETF de emergentes y de Argentina, dólar global, petróleo) | no mejora lo que ya da la macro argentina |

Hallazgo lateral: la **sobreventa** (RSI ≤ 30) sí precede rebotes (+1.1% a 5 ruedas, +1.5% a 14, +2% a 20
sobre el mercado, p ≤ 0.04), y la sobrecompra no precede bajas.

## Qué SÍ funcionó: macro argentina

Variables: riesgo país, dólar CCL / MEP / oficial / mayorista / blue y la brecha, tasa BADLAR, reservas y
base monetaria (todas diarias, con retrasos conservadores: 2 días las cotizaciones y 5 las del BCRA, que
publica con demora), más cercanía a elecciones.

| Horizonte | AUC solo técnicas | AUC con macro (ensamble) |
| --- | --- | --- |
| 20 ruedas | 0.54 | **0.60** [0.54, 0.66] |
| 30 ruedas | 0.50 | **0.63** |
| 40 ruedas | 0.51 | **0.66** |

Controles contra el autoengaño: **placebo** (macro desordenada en el tiempo → AUC vuelve a 0.53-0.55),
retrasos conservadores, prueba **prospectiva** (entrenar hasta 2024 y probar 2025-26 sin reentrenar),
y mejora en 6 de 7 años. El algoritmo importa poco: Extra Trees, boosting y una red densa dan 0.58-0.65;
las variables son lo que aporta.

### Validación del código de producción (`validar_modelo_macro.py`, 20 ruedas)

32.500 predicciones fuera de muestra 2020-2026, reentrenando cada ~6 meses:

- AUC **0.597**, IC95% [0.534, 0.658]; sobre 0.50 en 6 de 7 años (2021: 0.46).
- Señal **baja** (14% de los casos): acierta **56.7%** [46.9, 66.0] vs 39.5% de base.
- Señal **alza** (16% de los casos): acierta **71.9%** [59.1, 81.0] vs 60.5% de base.
- Neutral: 70% de los casos.

## Advertencias (importantes)

1. **La señal viene de niveles de variables lentas** (riesgo país, tasas, brecha). Sin esos niveles el
   AUC cae a 0.52-0.55. En 6 años hay pocos regímenes macro independientes (~3-4), así que la incertidumbre
   real es mayor que la del intervalo.
2. **Se eligieron variables y horizontes mirando los mismos datos**: hay sesgo a favor de lo encontrado.
   La única confirmación definitiva es el paper trading hacia adelante (`paper-trading.yml` ya lo registra).
3. **Hay que reentrenar seguido**: entrenado una sola vez en 2022 y probado en 2023-26, el AUC cae a 0.53-0.56.
4. La señal es **más fuerte en horizontes largos** (30-40 ruedas) pero no podría validarse en vivo antes de
   la entrega; por eso el modelo usa 20 ruedas (~1 mes).
5. En los últimos ~2 años (ventana de calibración fuera de muestra del entrenamiento actual) el AUC fue 0.54:
   el período reciente es más difícil que el promedio.

## Reproducir

- Datos: `data-colector` sirve las series (`/macro/argdatos/*`, `/interest-rate/ar/*`).
- `validar_modelo_macro.py`: valida el modelo tal como está en producción.
- `scripts_exploratorios/`: los scripts de la investigación (barrido de familias de variables, controles,
  placebo, ablaciones, familias de modelos, LSTM). **Apuntan a rutas de una carpeta de trabajo local**
  (`SP`/`R`/`ext/`): hay que ajustarlas para correrlos; sirven como registro de qué se probó y cómo.
- `../evaluacion_baselines.py` y `../evaluacion_modelos.py`: vara de comparación ("siempre sube", regla de RSI,
  logística) y comparación de modelos de la etapa anterior.
- `../mejora_modelos_existentes.patch`: cambios (más variables, horizonte 14, abstención) que se probaron sobre
  los modelos existentes y se dejaron fuera; el modelo `macro` los reemplaza como mejora.
