# Simulacion historica

Responde "si hubiera seguido las recomendaciones del sistema desde tal fecha, como le habria ido
a la cartera". Se para en una fecha pasada, entrena los modelos locales con datos hasta ese dia y
avanza rueda por rueda: reentrena como el cron de produccion, predice, rebalancea carteras y las
valoriza al cierre. Nunca usa datos posteriores al dia simulado.

Solo modelos locales de api_ml (LSTM, XGBoost, Transformer, ARIMA, macro, contexto). Los de Modal quedan
afuera. No toca `models/` ni ningun servicio de produccion: todo corre contra un snapshot en disco.

## Como correrla (copiar y pegar)

Todos los comandos van desde la carpeta de api_ml (la que tiene el `Makefile`).

### 1. Preparar el entorno (una sola vez)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install
```

En cada terminal nueva, antes de correr nada: `source .venv/bin/activate`. Si no existe
`python3.12`: `sudo apt install python3.12 python3.12-venv` (Ubuntu) o `sudo dnf install python3.12`
(Fedora). Con conda alcanza con un entorno de Python 3.12 activado y `make install`.

Para que el reporte salga en PDF hace falta LaTeX (sin LaTeX corre igual y deja solo el `.tex`):

```bash
# Ubuntu
sudo apt install texlive-latex-base texlive-latex-recommended texlive-lang-spanish
```

```bash
# Fedora
sudo dnf install texlive-latex texlive-babel texlive-babel-spanish texlive-geometry texlive-booktabs texlive-tools texlive-graphics texlive-float texlive-ec
```

### 2. Bajar los datos (una sola vez)

Si ya existe `simulacion/snapshot-collector/meta.json`, saltear este paso. Si no, levantar el
data-colector local (en su repo: `make network` la primera vez y `docker compose up -d --build`) y
bajar el snapshot (~10 anios de historia de cada ticker):

```bash
make snapshot-collector
```

### 3. Chequear que este todo

```bash
make simulacion-check
```

Tiene que terminar con `OK: estan todos los requisitos de la simulacion.` Si falta algo dice que y
como instalarlo (`apt` y `dnf`). `make simular` y `make simulacion` corren este mismo chequeo antes
de arrancar.

### 4. Correr la simulacion de un solo modelo

Entrena solo ese modelo, predice cada rueda del periodo y arma carteras simples (senal, histeresis,
cuantil 30%, inclinada) contra los benchmarks. Sin Black-Litterman ni consenso. Copiar la linea del
modelo que se quiera:

```bash
make simular MODELO=lstm NOMBRE=lstm-2026 DESDE=2026-01-02
```

```bash
make simular MODELO=xgboost NOMBRE=xgboost-2026 DESDE=2026-01-02
```

```bash
make simular MODELO=arima NOMBRE=arima-2026 DESDE=2026-01-02
```

```bash
make simular MODELO=macro NOMBRE=macro-2026 DESDE=2026-01-02
```

```bash
make simular MODELO=contexto NOMBRE=contexto-2026 DESDE=2026-01-02
```

```bash
make simular MODELO=ensamble_panel NOMBRE=ensamble-2026 DESDE=2026-01-02
```

`transformer_actual` es el Transformer que sirve la API (dolares, clasificacion, walk-forward; toma
el CCL y el riesgo pais del snapshot); `transformer` es el original (`CONFIG_ORIGINAL`). Tambien
estan los modelos de panel de la busqueda: `panel_gbm`, `ensamble_panel`, `panel_transformer`,
`reversion_panel`, `mercado`, `meta_reversion`, `dos_cabezas`.

### 5. (Opcional) Correr la simulacion completa

Todos los modelos juntos, con Black-Litterman, consenso y "mejor modelo por ticker". Tarda horas:

```bash
make simulacion NOMBRE=completa-2026 DESDE=2026-06-01 REENTRENO=semanal
```

### 6. Ver el resultado

Al terminar imprime donde quedo el reporte, por ejemplo:

```
Reporte LaTeX: /.../api_ml/simulacion/resultados/lstm-2026/reporte/reporte.tex
Reporte PDF:   /.../api_ml/simulacion/resultados/lstm-2026/reporte/reporte.pdf
```

Todo lo de la corrida queda en `simulacion/resultados/<NOMBRE>/` (ver "Archivos de una corrida" mas
abajo). Para rehacer el reporte de una corrida completa sin volver a simular:
`make simulacion-reporte NOMBRE=completa-2026`.

### Opciones

Se agregan al final del comando, p. ej.
`make simular MODELO=lstm NOMBRE=lstm-h1 DESDE=2026-01-02 HASTA=2026-06-30 REENTRENO=semanal`.

| Variable | Que hace | Default |
|---|---|---|
| `MODELO` | Modelo a simular (solo `simular`) | `ensamble_panel` |
| `NOMBRE` | Carpeta de la corrida en `simulacion/resultados/`. En `simular` tiene que ser nueva; en `simulacion`, si ya existe, retoma esa corrida | obligatorio |
| `DESDE` | Primera rueda simulada (`YYYY-MM-DD`); entrena con los datos anteriores | obligatorio |
| `HASTA` | Ultima rueda simulada | la ultima del snapshot |
| `REENTRENO` | `diario`, `semanal`, `mensual` o `nunca` | `mensual` en `simular`, `diario` en `simulacion` |
| `CONFIG` | JSON con campos de la config del modelo (solo `simular`), p. ej. `CONFIG='{"epochs": 20}'` | `{}` |
| `ARGS` | Cualquier otra opcion del script, p. ej. `ARGS='--comision 0.01 --rebalanceo 10'` | |
| `DISPOSITIVO` | `cuda` o `cpu` (solo `simular`) | `cuda` si hay GPU |
| `SNAPSHOT` | Carpeta del snapshot de datos | `simulacion/snapshot-collector` |
| `PYTHON` | Python a usar | `python` |

### Si algo falla

- `... ya existe: usar otro --nombre` (en `simular`): ya hay una corrida con ese `NOMBRE`; usar otro
  (o borrar `simulacion/resultados/<NOMBRE>/`).
- `ERROR: No existe el snapshot de datos ...`: falta el paso 2.
- `ERROR: Falta instalar el paquete de Python ...`: falta activar el venv (`source .venv/bin/activate`)
  o el paso 1.
- `invalid choice` en `--modelo`: el nombre del modelo esta mal escrito; la lista sale en el error.
- `(sin PDF: falta pdflatex o fallo la compilacion ...)`: el `.tex` esta igual; instalar LaTeX
  (paso 1) y correr de nuevo, o subir la carpeta `reporte/` a Overleaf.

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

Para probar un modelo (o una variante de la busqueda) sin esperar al simulador
completo: entrena solo ese modelo con datos anteriores a `--desde` (reentreno
`mensual` por default, o `semanal` / `diario` / `nunca`), predice cada rueda, mide la direccion
en dolares con intervalos contra la reversion y las referencias (`significancia`) y arma carteras
simples (senal, histeresis, cuantil 30%, inclinada y benchmarks). Sin Black-Litterman, consenso
ni backtests en vivo.

Sirve cualquier modelo de `busqueda.MODELOS` (`--modelo`): los de panel y tambien los de produccion
que predicen ticker por ticker (`lstm`, `xgboost`, `arima`, `macro`, `contexto`), adaptados en
`por_ticker`. Macro y contexto se entrenan como en el simulador completo y predicen con la macro /
el mercado del dia simulado, pero sin el piso de AUC: cada reentreno reemplaza al anterior.

```bash
make simular MODELO=ensamble_panel NOMBRE=ens-2025 DESDE=2025-01-02 HASTA=2025-12-30
make simular MODELO=panel_transformer NOMBRE=pt-2025 DESDE=2025-01-02 CONFIG='{"n_semillas": 1}' REENTRENO=nunca
make simular MODELO=xgboost NOMBRE=xgb-2026 DESDE=2026-01-02
# Otra comision / rebalanceo sin volver a entrenar
python -m src.simulacion.un_modelo --nombre ens-2025 --solo-analisis --comision 0.01
```

Deja en `simulacion/resultados/<nombre>/` `resumen.md` (entrenamientos, direccion y carteras),
`entrenamientos.jsonl` (lo que devolvio cada `fit`: sus metricas de validacion),
`predicciones.jsonl`, `direccion.csv`, `carteras.csv`, los `.jsonl` de carteras y
`reporte/reporte.tex` (y `reporte.pdf` si hay `pdflatex`). `--config` acepta tambien `"modelo"` e
`"historia"`, asi que se puede pegar una variante de los planes de la busqueda.

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
