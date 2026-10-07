# Modelo macro: resultados

Predice si una acción sube o baja a 20 ruedas usando variables macro de Argentina (riesgo país, dólar CCL/MEP/oficial y brecha, BADLAR, reservas, base monetaria), indicadores técnicos y cercanía a elecciones. Ensamble de boosting y Extra Trees.

## Cómo se midió

Walk-forward 2020-2026 sobre 20 acciones del Merval: el modelo se entrena solo con datos anteriores, se prueba en el tramo siguiente y se reentrena cada ~6 meses. Intervalos de confianza por bloques de mes.

## Resultados (20 ruedas, 32.500 predicciones)

| Medida | Resultado |
| --- | --- |
| Puntaje general (AUC) | 0.597 (intervalo 0.53-0.66); sobre 0.50 en 6 de 7 años |
| Acierto de dirección sobre todas las predicciones | 59% a 62% (decir "sube" siempre: 60.5%) |
| Señal `baja` (14% de los casos) | 56.7% de acierto (base 39.5%) |
| Señal `alza` (16% de los casos) | 71.9% de acierto (base 60.5%) |
| Retorno en 20 ruedas, quinto más alcista vs más bajista | +13.9% vs +3.2% (diferencia 10.7 puntos, intervalo 3.7-17.4) |

En el 70% restante responde `neutral`.

## Qué aportó y qué no

- **Aportó:** la macro argentina. Sin ella, los indicadores técnicos solos no distinguen subas de bajas.
- **No aportó:** mercado global (S&P 500, VIX, ETF de Argentina), brecha contra el ADR, LSTM sobre ventanas de precios, entrenar por períodos cortos o por régimen.
- El plazo de 20 ruedas se eligió por ser un mes y poder validarse en vivo; a 30-40 ruedas el puntaje es mayor (0.63 y 0.66).

## Límites

- La señal depende de niveles de variables lentas (riesgo país, tasas, brecha) y en 6 años hay pocos regímenes macro distintos.
- Variables y plazo se eligieron mirando los mismos datos: la confirmación es el paper trading.
- Hay que reentrenarlo periódicamente (workflow mensual).
- `baja` es menos precisa que `alza`.
