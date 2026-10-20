# Simulacion historica

Responde "si hubiera seguido las recomendaciones del sistema desde tal fecha, como le habria ido
a la cartera". Se para en una fecha pasada, entrena los modelos locales con datos hasta ese dia y
avanza rueda por rueda: reentrena como el cron de produccion, predice, rebalancea carteras y las
valoriza al cierre. Nunca usa datos posteriores al dia simulado.

Solo modelos locales de api_ml (LSTM, XGBoost, Transformer, ARIMA, macro, contexto). Los de Modal quedan
afuera. No toca `models/` ni ningun servicio de produccion: todo corre contra un snapshot en disco.

## Pasos

```bash
# 1. Bajar los datos una vez (Yahoo + data-colector) a simulacion/snapshot/
python -m src.simulacion.snapshot

# 2. Correr la simulacion (se puede cortar y retomar; con un --hasta posterior la extiende)
python -m src.simulacion.simulador --nombre desde-2026-06-semanal --desde 2026-06-01 --reentreno semanal

# 3. Reporte: resumen.json, reporte.md y graficos/ (no vuelve a correr nada)
python -m src.simulacion.reporte --nombre desde-2026-06-semanal

# Opcional: rehacer las carteras con otras reglas, sin reentrenar
python -m src.simulacion.rearmar --nombre desde-2026-06-semanal --salida comision-1pct --comision 0.01
python -m src.simulacion.reporte --nombre desde-2026-06-semanal --rearmado comision-1pct
```

En PC-Escritorio, con la imagen de api_ml:

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp -v "$PWD":/app -w /app \
  api_ml-service_api_ml:latest python -m src.simulacion.simulador --nombre ... --desde ...
```

## Simulacion rapida de un solo modelo

Para probar una variante de un modelo de panel (p. ej. el Transformer V2) sin esperar al
simulador completo: entrena solo ese modelo con datos anteriores a `--desde` (reentreno
`mensual` por default, o `semanal` / `diario` / `nunca`), predice cada rueda, mide la direccion
en dolares con intervalos contra la reversion y las referencias (`significancia`) y arma carteras
simples (senal, histeresis, cuantil 30%, inclinada y benchmarks). Sin Black-Litterman, consenso,
ARIMA, macro ni backtests en vivo.

```bash
make simular-v2 SIM_NOMBRE=v2-2025 SIM_DESDE=2025-01-02 ARGS='--hasta 2025-12-30'
make simular-v2 SIM_NOMBRE=v2-bce SIM_CONFIG='{"perdida": "bce", "n_semillas": 1}' ARGS='--reentreno nunca'
# Otra comision / rebalanceo sin volver a entrenar
python -m src.simulacion.un_modelo --nombre v2-2025 --solo-analisis --comision 0.01
```

Deja en `simulacion/resultados/<nombre>/` `resumen.md` (entrenamientos, direccion y carteras),
`entrenamientos.jsonl` (lo que devolvio cada `fit`: las metricas de validacion del V2),
`predicciones.jsonl`, `direccion.csv`, `carteras.csv` y los `.jsonl` de carteras. Cualquier modelo
de `busqueda.MODELOS` sirve (`--modelo`); `--config` acepta tambien `"modelo"` e `"historia"`,
asi que se puede pegar una variante de los planes de la busqueda.

## Que replica de produccion

| Pieza | Como en produccion |
|---|---|
| Reentreno LSTM / XGBoost / Transformer | `promote_if_better` (`src/retrain_guard.py`), con el sidecar `.score.json` del vigente. `--reentreno diario` (como `retrain.yml`), `semanal` (ultima rueda de cada semana) o `mensual` (el mismo dia que macro y contexto: ~la mitad de tiempo que semanal) |
| Reentreno macro | El 1 de cada mes, se guarda si el AUC fuera de muestra llega a `DEFAULT_MIN_AUC` |
| Reentreno contexto | Igual que el macro (`retrain-contexto.yml`, AUC minimo de `src/train_context.py`), con el historico de los lideres del snapshot en vez de bajarlo de Yahoo |
| ARIMA | Se ajusta por ticker en cada prediccion |
| Backtest en vivo | `backtest_predict_df` (ultimas 60 ruedas, paso 5). Macro y contexto, solo sobre predicciones con vista (`backtest_sin_neutrales`) |
| Consenso | `compute_consensus` con el backtest solo de LSTM / XGBoost (`_LIVE_BACKTEST_MODELS`) |
| Black-Litterman | `entry` con cartera ancla equal weight, GARCH en la diagonal y universo panel lider |

## Carteras

Todas parten de `--capital` (default $1.000.000), pagan `--comision` (default 0,5%) por operacion,
se rebalancean al cierre cada `--rebalanceo` ruedas (default 5) y solo invierten en el panel lider.
Lo que no se asigna queda en efectivo, sin interes.

- `senal_<modelo>`: partes iguales entre los tickers donde el modelo da "alza".
- `bl_<modelo>_<perfil>`: pesos de Black-Litterman con las predicciones del modelo.
- `consenso_<perfil>`: partes iguales entre los tickers con consenso de compra.
- `senal_mejor_<seleccion>` y `bl_mejor_<seleccion>_<perfil>`: en cada ticker sigue al modelo con
  mejor accuracy del backtest en vivo, ajustada por cantidad de casos (mismo criterio que
  `pick_best_model` del back). `prod` elige entre LSTM y XGBoost; `todos`, entre los cinco.
- Benchmarks: equal weight con rebalanceo, equal weight buy & hold y Merval.

## Archivos de una corrida (`simulacion/resultados/<nombre>/`)

| Archivo | Contenido |
|---|---|
| `config.json`, `estado.json` | Parametros y estado para retomar |
| `predicciones.jsonl` | Cada prediccion de cada modelo, ticker y rueda |
| `backtests.jsonl` | Backtest en vivo por modelo y ticker en los dias de rebalanceo |
| `elecciones.jsonl` | Modelo elegido por ticker en cada rebalanceo |
| `promociones.jsonl` | Cada reentreno: score del candidato, del vigente y si se promovio |
| `pesos.jsonl`, `operaciones.jsonl`, `valores.jsonl` | Pesos objetivo, operaciones y valor diario de cada cartera (con CCL) |
| `errores.jsonl` | Rebalanceos de Black-Litterman que fallaron (la cartera queda como estaba) |
| `artefactos/versiones/` | Cada version promovida de cada modelo, por fecha |
| `movimientos.csv` | Cada compra / venta con la estrategia y el modelo de la cartera, el peso objetivo, la prediccion del modelo que la decidio, la senal de todos los modelos para ese ticker ese dia y el retorno real a 5 y 20 ruedas (`src/simulacion/movimientos.py`) |
| `rearmados/sin-comision/` | Las mismas carteras sin comision (se rearman solas al terminar), con su `movimientos.csv`; el reporte compara las dos |

Las metricas y el reporte salen de estos archivos, asi que cambiarlos no obliga a volver a
correr. `rearmar` rehace las carteras a partir de `predicciones.jsonl` y `backtests.jsonl`; con
los mismos parametros reproduce `valores.jsonl` exacto. Como el backtest se guarda solo en los
dias de rebalanceo, el `--rebalanceo` de un rearmado tiene que ser multiplo del original.

## Limitaciones

Estan listadas al final de cada `reporte.md`: ejecucion al cierre, efectivo sin interes, sesgo de
supervivencia del universo, macro entrenado con el historico del snapshot, consenso sin los
modelos de Modal, horizontes mezclados en Black-Litterman "todos" y un unico periodo de mercado.
