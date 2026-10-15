# Exploración para predecir mejor (2026-10-15)

Panel de 35 acciones BYMA, 2011-2026 (117 mil filas), datos diarios de Yahoo + macro de ArgentinaDatos/BCRA. Todo con walk-forward anual: se entrena con datos anteriores al año de prueba (con purga del horizonte) y se evalúa en ese año, 2016-2026. Los intervalos son bootstrap por bloques de mes. Scripts en esta carpeta (`dl.py` baja datos, `feat.py`/`feat2.py` arman features, `ev.py` evalúa, `e1`…`e10` son los experimentos en orden).

La vara de comparación no es el 50 %: es la tasa base del mismo año (cuánto acertaría decir siempre «sube» o siempre «baja»).

## Conclusión corta

1. **Los modelos de producción miden ~azar porque les falta el contexto del mercado**, no por el algoritmo. Con las features que usan hoy (retorno, volumen, RSI, SMA, momentum) el AUC es 0,507-0,516. Agregando contexto de mercado y un ensamble sube a 0,539 (5 ruedas) y 0,555 (20 ruedas), con intervalo que no toca 0,5.
2. **La mejora es real pero modesta.** No hay forma de «predecir bien» en sentido absoluto: el acierto sobre todas las predicciones sigue cerca de la tasa base. Lo que sí aparece es ventaja cuando el modelo se anima a opinar (extremos).
3. **El timing del mercado no se predice** con nada de lo probado (técnico, FX, macro Argentina, global): AUC 0,48-0,52 sobre 2011-2026. La ventaja viene de comparar acciones entre sí (cuál va a rendir más que el resto).
4. **La macro no aguanta una validación más larga.** El modelo `macro` mostró AUC 0,60 en 2020-2026; sobre 2016-2026 con más acciones y años, agregar macro/FX a las features técnicas empeora o no cambia (AUC 0,52 vs 0,556 sin macro a 20 ruedas). Los pocos regímenes macro hacen que esa señal sea frágil. Conviene tratarlo como hipótesis a confirmar con paper trading, no como la mejora principal.

## Qué se probó y qué pasó (20 ruedas, etiqueta «sube», AUC pooled)

| Experimento | AUC | Lectura |
| --- | ---: | --- |
| Solo técnicas (como producción) | 0,528 | casi azar |
| + contexto de mercado (canasta de líderes, fuerza relativa, amplitud) | **0,556** | única adición que ayuda |
| + FX/brecha | 0,547 | sin mejora |
| + macro Argentina | 0,521 | empeora |
| + global (S&P, VIX, DXY, petróleo…) | 0,538 | sin mejora |
| + ADR (gap local vs ADR) | 0,541 | pequeña, dentro del ruido |
| Features rankeadas por fecha | 0,523 | peor |
| Regresión logística | 0,527 | peor: hacen falta árboles |
| Árboles más profundos / más superficiales | 0,549 / 0,551 | igual |
| Ventana móvil 3 y 5 años | 0,551 | igual |
| Etiqueta relativa (rinde más que el resto) | 0,525 [0,514-0,537] | señal chica pero muy estable |
| Regresión sobre retorno ajustado por volatilidad | 0,538 | peor |
| Ensamble de 6 árboles con peso decreciente (3 años) | 0,551-0,558 | levemente más estable |
| + peers (retorno de las 5 acciones más correlacionadas), dispersión, rankings, residuo vs mercado | 0,544-0,555 | sin mejora clara |

Horizonte: 5 ruedas AUC 0,537; 10 ruedas 0,539; 20 ruedas 0,556. Predecir el retorno en USD (CCL) en vez de pesos: 0,532.

## Ranking entre acciones (la señal más consistente)

Correlación de rango diaria entre el puntaje del modelo y el retorno posterior (IC), solo con técnicas + mercado: +0,075 a 5 ruedas y +0,051 a 20 ruedas, positiva en todos los años 2016-2026 y en 79 % de los meses. Subuniverso de acciones líquidas: +0,062. La señal baja pero no desaparece al entrar con 1-3 días de rezago (0,067 → 0,042), así que no es solo rebote de precio.

Ojo: agregar solo un indicador de «tiene ADR» sube el IC de 0,075 a 0,090. Eso es identidad de la acción (empresas grandes que sobrevivieron), no información nueva, así que se descarta como señal. El aporte real del ADR, dentro de las acciones que lo tienen, es de +0,02 de IC.

## Qué puede decir la web con esto (ensamble + contexto, acciones líquidas, vs tasa base del mismo año)

| Horizonte | Señal | Acierta | Tasa base | Ventaja (IC 95 %) |
| --- | --- | ---: | ---: | ---: |
| 5 ruedas | alza (10 % más alto) | 63-64 % | 53,6 % | +10 pp (+4 a +15) |
| 5 ruedas | baja (10 % más bajo) | 52-53 % | 46,4 % | +10 pp (+5 a +17) |
| 20 ruedas | alza (10 % más alto) | 66-69 % | 59,2 % | +7 pp (+1 a +14) |
| 20 ruedas | baja (10 % más bajo) | 52 % | 40,8 % | +12 pp (+6 a +18) |

El mismo modelo con las features de producción: 0,507 de AUC y la señal de baja casi sin ventaja (+1 a +5 pp, intervalo toca 0).

## Límites que no se pueden ignorar

- **Sesgo de supervivencia:** el universo son acciones que existen hoy. Eso infla las señales de «comprar la caída» y de identidad. No hay datos de acciones deslistadas para corregirlo.
- **En 2024-2026 los intervalos se ensanchan** (alza +4,7 pp [-3,9 a +14,8] a 5 ruedas): la ventaja no está confirmada en el tramo más reciente. Hay que verla en paper trading.
- La baja absoluta sigue en ~52 %: «va a bajar» acierta poco más que una moneda aunque supere su tasa base.
- Los umbrales de «extremo» se calcularon sobre las predicciones del año de prueba; para producción hay que fijarlos con validación previa.
- Probé más de 40 variantes sobre los mismos datos. Una ventaja de 1-2 pp entre variantes es ruido; solo el contexto de mercado es una diferencia clara.
- Regla diaria sin calibrar (top/bottom 20 % de cada día): ventaja menor, +3 pp a 5 ruedas (t≈2,5) y +1 a +3 pp a 20 ruedas (no significativa).

## Qué hacer con esto (en orden de valor/costo)

1. Agregar contexto de mercado a los modelos existentes (canasta de líderes, retorno relativo a 5/20/60 ruedas, amplitud, beta) y usar ensamble con peso decreciente. Es el único cambio con respaldo. Ya hay una versión en `notebooks/mejora_modelos_existentes.patch`.
2. Que la señal use abstención calibrada con validación previa: opinar solo en el ~10-25 % más extremo, neutral el resto, y mostrar la ventaja contra la tasa base, no el acierto bruto.
3. Presentar el resultado como ranking relativo («destaca/rezagada frente al mercado») además de alza/baja, que es lo que el modelo sabe hacer.
4. Dejar el modelo `macro` marcado como experimental hasta tener meses de paper trading.
5. No invertir más en LSTM/Transformer: no aportan sobre árboles con estas features.

## Implementación y validación con el código real (modelo `contexto`)

Implementado en `src/context_trend.py` y validado con `validar_modelo_contexto.py`: por cada año 2016-2026 se arma un modelo solo con datos anteriores (con purga), con la misma calibración de umbrales que el entrenamiento real, sobre las 20 acciones líquidas del universo y datos diarios corregidos (ver abajo). Ventaja = acierto de la señal menos la tasa base del mismo año; intervalo por bootstrap de meses.

| Variante | AUC | Señal de alza | Señal de baja |
| --- | ---: | --- | --- |
| Solo técnicas (40 variables, ensamble) | 0,526 | 69,0 % vs 60,0 %: **+8,9 pp [+3,8, +14,1]** | +7,0 pp [-2,8, +16,5] (5 % de los casos) |
| **Con contexto de mercado (`contexto`)** | 0,538 | 71,1 % vs 60,0 %: **+11,1 pp [+5,7, +15,8]** | -2,8 pp [-12,7, +8,9]: sin ventaja |
| Modelos de producción previos (~6 variables, sin contexto, exploración) | 0,507-0,516 | ~azar | ~azar |

Lectura honesta:
- La señal de **alza** del modelo `contexto` le gana a la tasa base de forma significativa (el intervalo no toca 0) y se mantiene en 2024-2026 (+9,8 pp [+1,2, +16,6]).
- La señal de **baja no tiene ventaja demostrada**. Por eso el modelo la apaga sola cuando fuera de muestra no supera a su tasa base (hoy está apagada en el artefacto entrenado: acierta 37,8 % contra 38,9 % de base).
- El **contexto en sí suma poco** frente a las técnicas ampliadas (+2 pp en alza, dentro del ruido). La mejora frente a producción viene sobre todo de usar más variables técnicas, ensamblar, ponderar lo reciente y abstenerse; el contexto es el componente que la exploración mostró más consistente, pero su aporte marginal no es concluyente.
- El acierto absoluto de «alza» está muy asociado a que el mercado subió en el período (tasa base 60 %). Sigue pendiente confirmarlo con paper trading (el modelo ya está en `paper-trading.yml`).

## Bug de datos encontrado y corregido

`src/data.py::fetch_history_yahoo` pedía `range=max` cuando `days > 1825` y Yahoo, en ese caso, ignora `interval=1d` y devuelve **velas mensuales** (~320 filas en 25 años). Todo entrenamiento con `--days 4000` (incluido el del modelo `macro`) recibía meses en lugar de ruedas diarias. Ahora se piden fechas explícitas (`period1/period2`) y se obtienen 4000 ruedas diarias reales. **El modelo `macro` (y los artefactos entrenados con `--days` grande) hay que reentrenarlos.**
