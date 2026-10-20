"""Transformer de tendencia, version 2: el Transformer original con sus correcciones.

Objetivo: predecir si cada accion va a **subir o bajar** en las proximas
``horizonte`` ruedas, medido en dolares CCL.

El original (``src.transformer``) mira cada accion sola, en pesos, con 3
features (retorno, cambio de volumen y rango) normalizadas con una media y un
desvio globales, y predice el retorno log a 5 ruedas con MSE. Con eso aprende
sobre todo la deriva de la inflacion (dice "alza" casi siempre) y no ve la
reversion de corto plazo. Esta version corrige:

1. **Datos**: precios del collector (OHLC ajustados por dividendos) en
   dolares CCL (``moneda="usd"``: sin la deriva de la inflacion). Dentro del
   modelo, en ``fit`` y en ``predict_panel``, se corrigen los saltos que la
   fuente no ajusto (splits, canjes; ``limpiar_saltos``, ver
   ``src.simulacion.limpieza``). Cuanta historia usa lo decide quien lo
   entrena (``--days`` de ``train_transformer_v2``).
2. **Features** de ``src.panel_features``: retornos de 1 a 60 ruedas,
   reversion relativa al panel y al sector, volatilidades, volumen anormal,
   distancias a medias, drawdowns y beta; retornos divididos por la vol del
   propio ticker y rankeados entre las acciones del dia. Con ``macro=True``,
   ademas un token con el contexto del mercado entero (CCL, brecha, riesgo
   pais, BADLAR, reservas, eventos; con sus retrasos de publicacion): la
   direccion de una accion depende sobre todo de la del mercado.
3. **Target**: retorno en dolares a ``horizonte`` ruedas en unidades de vol del
   ticker, recortado, **con** la media del dia restada (``"sin_media"``, default):
   el modelo ordena las acciones contra el resto del panel. Con ``"absoluto"``
   conserva si el mercado entero sube o baja, pero la red no lo predice y
   colapsa a decir "alza" casi siempre (ver ``notas/informe-transformer-v2.md``).
4. **Red**: la de ``src.panel_transformer`` (encoder temporal por accion con
   token CLS y atencion entre las acciones del mismo dia, que le muestra al
   modelo como se esta moviendo el resto del panel). Sin el
   ``* sqrt(d_model)`` del original, que tapaba el embedding posicional.
5. **Entrenamiento**: perdida Huber sobre el target (robusta a dias extremos)
   o ``"bce"``: clasificacion sube / baja (entropia cruzada sobre el signo del
   target), que optimiza lo mismo que se mide. Pesos por antiguedad, early
   stopping por **acierto balanceado de la direccion** en una validacion con
   gap = horizonte, y despues **reentreno con todos los datos**
   (``reentrenar_completo``) por la cantidad de epocas elegida: el original
   deployaba un modelo que nunca vio el ultimo 15% de la historia.
   ``n_semillas`` redes se promedian.
6. **Senal**: "alza" o "baja" segun el signo del retorno predicho en dolares.
   Antes, dos ajustes opcionales del score, calibrados al entrenar con las
   predicciones de las redes finales sobre los dias de entrenamiento (en
   ``metrics``):

   - ``peso_reversion``: ensamble con la regla de reversion (lo contrario del
     retorno en dolares de las ultimas 5 ruedas, ``ret_5_nivel``), las dos
     partes estandarizadas por su desvio en entrenamiento. La regla sola le
     gano al original y al primer V2 en direccion.
   - ``centrar_senal``: le resta al score su mediana de entrenamiento. Sin eso,
     la red aprende la deriva del mercado de su historia y dice casi siempre lo
     mismo (el primer V2 decia "alza" el 73% de las veces).

   Con ``cobertura_senal`` (p. ej. 0,2) solo se da senal en esa fraccion de
   las predicciones mas fuertes y el resto queda "neutral" (umbral calibrado
   igual). ``predicted_close`` es el cierre en pesos esperado si el CCL no se
   mueve (con ``"bce"``: el desvio esperado del ticker por ``2p - 1``, siendo
   ``p`` la probabilidad de suba).

Los defaults (target relativo, seleccion por la perdida de validacion, BCE y
la red de abajo) son la configuracion elegida por la busqueda de Optuna
(``src.simulacion.optuna_v2``, corrida ``optuna-v2``: 76 trials, objetivo IC
sobre la regla de reversion en 8 folds 2019-2023, top 5 reevaluado con otras
semillas), con ``HISTORIA_V2`` ruedas de historia. Ninguna configuracion le
gano a la regla de reversion (PBO 91%): estos valores son mejores que los
anteriores (``CONFIG_ANTERIOR``, el V2 absoluto colapsaba a "alza") pero no que
el ensamble lineal. Lo que mas pesa es la regularizacion (weight decay, lr y
dropout); la arquitectura casi no cambia el resultado.

Entrenamiento: ``make train-transformer-v2`` (``src.train_transformer_v2``).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.errors import NotEnoughDataError
from src.panel_features import NOMBRES, Normalizador, particion, pesos_temporales
from src.panel_gbm import features_y_contexto
from src.panel_transformer import (
    MIN_DIAS,
    PERDIDAS,
    PanelTransformerConfig,
    PanelTransformerModel,
    _RedPanel,
)
from src.simulacion.limpieza import limpiar_precios
from src.trend_common import senal

# Retorno en dolares de las ultimas 5 ruedas en unidades de vol, sin rankear.
_RET_5 = NOMBRES.index("ret_5_nivel")


# Ruedas de historia para entrenar (todo el snapshot del collector, ~10 anios).
HISTORIA_V2 = 2400
# Defaults hasta la busqueda de Optuna (V2 absoluto): los usan los planes viejos
# de ``src.simulacion.busqueda`` para que sigan midiendo lo mismo.
CONFIG_ANTERIOR = {
    "target": "absoluto",
    "seleccion": "direccion",
    "perdida": "huber",
    "d_model": 32,
    "dim_feedforward": 64,
    "capas_temporales": 1,
    "weight_decay": 0.3,
    "learning_rate": 1e-4,
}


@dataclass(frozen=True)
class TransformerV2Config(PanelTransformerConfig):
    # Objetivo y red: los elegidos por Optuna (ver docstring del modulo)
    target: str = "sin_media"
    seleccion: str = "perdida"
    perdida: str = "bce"
    ventana: int = 1
    d_model: int = 16
    nhead: int = 4
    dim_feedforward: int = 32
    capas_temporales: int = 2
    capas_cruzadas: int = 1
    dropout: float = 0.3
    weight_decay: float = 1.4e-3
    learning_rate: float = 1.2e-4
    n_semillas: int = 5
    # Correcciones propias de esta version
    limpiar_saltos: bool = True
    reentrenar_completo: bool = True
    # Ajustes de la senal (ver docstring del modulo)
    peso_reversion: float = 0.0
    centrar_senal: bool = False
    # Fraccion de predicciones (las mas fuertes) con senal; None = todas.
    cobertura_senal: float | None = None

    def perdidas_validas(self) -> tuple[str, ...]:
        return (*PERDIDAS, "bce")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cobertura_senal is not None and not 0 < self.cobertura_senal <= 1:
            raise ValueError("cobertura_senal tiene que estar entre 0 y 1")
        if not 0 <= self.peso_reversion <= 1:
            raise ValueError("peso_reversion tiene que estar entre 0 y 1")


class TransformerV2Model(PanelTransformerModel):
    ARQUITECTURA = "transformer_v2"
    CONFIG = TransformerV2Config

    @property
    def version(self) -> str:
        return f"transformer-v2-{self.config.moneda}"

    def _preparar(self, histories: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Historias sin saltos de splits / canjes (si ``limpiar_saltos``)."""
        histories = {s: df for s, df in histories.items() if not df.empty}
        if self.config.limpiar_saltos:
            histories = limpiar_precios(histories)[0]
        return histories

    # ---------------------------------------------------------- entrenamiento
    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        cfg = self.config
        histories = self._preparar(histories)
        fp, ctx = features_y_contexto(histories, cfg, self.proveedor)
        dias = self._dias_muestra(fp)
        if len(dias) < MIN_DIAS:
            raise NotEnoughDataError(f"muy pocos dias para entrenar ({len(dias)})")
        tr, va = particion(fp.fechas[dias], cfg.val_fraction, gap=cfg.horizonte)
        dias_tr, dias_va = dias[tr], dias[va]

        # 1) Eleccion de epocas: normalizadores y entrenamiento solo con train.
        datos = self._datos(fp, ctx, hasta=dias_tr[-1] + 1)
        n_features = datos["x"].shape[-1]
        self._redes, por_red = [], []
        for k in range(cfg.n_semillas):
            red, info = self._entrenar_red(cfg.seed + k, datos, dias_tr, dias_va, n_features)
            self._redes.append(red)
            por_red.append(info)

        # 2) Reentreno con todos los dias (train, gap y validacion) por la misma
        # cantidad de epocas, con normalizadores ajustados sobre toda la historia.
        if cfg.reentrenar_completo:
            datos = self._datos(fp, ctx, hasta=len(fp.fechas))
            self._redes = [
                self._entrenar_fijo(cfg.seed + k, datos, dias, n_features, info["epoca_mejor"])
                for k, info in enumerate(por_red)
            ]

        self.trained_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.metrics = {
            "dias_train": int(len(dias) if cfg.reentrenar_completo else len(dias_tr)),
            "dias_val": int(len(dias_va)),
            f"val_{cfg.seleccion}": round(float(np.mean([i["val"] for i in por_red])), 4),
            "epoca_mejor": [i["epoca_mejor"] for i in por_red],
        }
        self._calibrar(fp, datos, dias)
        return self.metrics

    def _datos(self, fp, ctx: pd.DataFrame, hasta: int) -> dict:
        """Tensores de entrenamiento, con los normalizadores ajustados sobre las
        fechas ``[:hasta]`` (las ventanas miran hacia atras)."""
        cfg = self.config
        self._normalizador = Normalizador().fit(fp.valores[:hasta], fp.mascara[:hasta])
        if cfg.macro:
            self._columnas_macro = list(ctx.columns)
            self._normalizador_macro = Normalizador().fit(ctx.to_numpy(dtype=float)[:hasta])
        x, tickers, sectores, macro = self._tensores(fp, ctx)
        disp = torch.device(self.dispositivo)
        return {
            "x": torch.tensor(x, device=disp),
            "presentes": torch.tensor(fp.mascara, device=disp),
            "tickers": torch.tensor(tickers, device=disp),
            "sectores": torch.tensor(sectores, device=disp),
            "macro": None
            if macro is None
            else torch.tensor(macro, dtype=torch.float32, device=disp),
            "y": torch.tensor(np.nan_to_num(fp.target), dtype=torch.float32, device=disp),
            "m": torch.tensor(fp.mascara_target, device=disp),
            "w": torch.tensor(
                pesos_temporales(fp.fechas, cfg.vida_media_anios), dtype=torch.float32, device=disp
            ),
        }

    def _perdida(self, score: torch.Tensor, datos: dict, dias: torch.Tensor) -> torch.Tensor:
        if self.config.perdida != "bce":
            return super()._perdida(score, datos, dias)
        # Clasificacion: el score es el logit de "sube" (target > 0).
        y, m, w = datos["y"][dias], datos["m"][dias], datos["w"][dias]
        bce = F.binary_cross_entropy_with_logits(score, (y > 0).float(), reduction="none") * m
        por_dia = bce.sum(1) / m.sum(1).clamp(min=1)
        return (por_dia * w).sum() / w.sum()

    def _entrenar_fijo(
        self, seed: int, datos: dict, dias: np.ndarray, n_features: int, epocas: int
    ):
        """Entrena una red nueva con ``dias`` durante ``epocas``, con el mismo
        learning rate por epoca que tuvo la corrida con early stopping."""
        cfg = self.config
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        disp = datos["x"].device
        n_macro = 0 if datos["macro"] is None else datos["macro"].shape[1]
        red = _RedPanel(cfg, n_features, n_macro, len(self.universo)).to(disp)
        optimizador = torch.optim.AdamW(
            red.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        def lr_lambda(epoca: int) -> float:
            if cfg.warmup_epocas > 0 and epoca < cfg.warmup_epocas:
                return (epoca + 1) / cfg.warmup_epocas
            progreso = (epoca - cfg.warmup_epocas) / max(1, cfg.epocas - cfg.warmup_epocas)
            return 0.5 * (1 + math.cos(math.pi * min(progreso, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizador, lr_lambda)
        for _ in range(max(1, epocas)):
            red.train()
            orden = rng.permutation(dias)
            for i in range(0, len(orden), cfg.batch_dias):
                lote = torch.tensor(orden[i : i + cfg.batch_dias], device=disp)
                optimizador.zero_grad()
                perdida = self._perdida(self._forward(red, datos, lote), datos, lote)
                perdida.backward()
                nn.utils.clip_grad_norm_(red.parameters(), max_norm=1.0)
                optimizador.step()
            scheduler.step()
        red.eval()
        return red.cpu()

    def _scores_entrenamiento(
        self, fp, datos: dict, dias: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score (promedio de las redes finales), reversion y vol de las celdas con
        target de ``dias``."""
        disp = datos["x"].device
        dias_t = torch.tensor(dias, device=disp)
        scores = []
        with torch.no_grad():
            for red in self._redes:
                red.to(disp)
                lotes = [
                    self._forward(red, datos, dias_t[i : i + 64]) for i in range(0, len(dias_t), 64)
                ]
                scores.append(torch.cat(lotes).cpu().numpy())
                red.cpu()
        m = fp.mascara_target[dias]
        score = np.mean(scores, axis=0)
        return score[m], -fp.valores[dias][..., _RET_5][m], fp.vol[dias][m]

    def _calibrar(self, fp, datos: dict, dias: np.ndarray) -> None:
        """Escalas del ensamble con la reversion, centro y umbral de la senal (en
        ``metrics``), con las predicciones sobre los dias de entrenamiento. Solo
        se usa su escala y su sesgo, no si acertaron."""
        cfg = self.config
        if not (cfg.peso_reversion or cfg.centrar_senal or cfg.cobertura_senal):
            return
        score, reversion, vol = self._scores_entrenamiento(fp, datos, dias)
        if cfg.peso_reversion:
            self.metrics["escala_modelo"] = float(np.std(score)) or 1.0
            self.metrics["escala_reversion"] = float(np.std(reversion)) or 1.0
        combinado = self._combinar(score, reversion)
        if cfg.centrar_senal:
            self.metrics["centro_senal"] = float(np.median(combinado))
            combinado = combinado - self.metrics["centro_senal"]
        if cfg.cobertura_senal:
            retornos = [
                self._a_log_return(float(c), float(v)) for c, v in zip(combinado, vol, strict=True)
            ]
            umbral = float(np.quantile(np.abs(retornos), 1 - cfg.cobertura_senal))
            self.metrics["umbral_senal"] = umbral

    def _combinar(self, score: np.ndarray, reversion: np.ndarray) -> np.ndarray:
        """Ensamble con la reversion, en la escala del score del modelo."""
        peso = self.config.peso_reversion
        if not peso:
            return score
        modelo, rev = self.metrics["escala_modelo"], self.metrics["escala_reversion"]
        return ((1 - peso) * score / modelo + peso * reversion / rev) * modelo

    # -------------------------------------------------------------- prediccion
    def _ajustar_scores(self, scores: np.ndarray, fp, dia: int) -> np.ndarray:
        combinado = self._combinar(scores, -fp.valores[dia, :, _RET_5])
        return combinado - self.metrics.get("centro_senal", 0.0)

    def _a_log_return(self, score: float, vol: float) -> float:
        if self.config.perdida != "bce":
            return super()._a_log_return(score, vol)
        # Logit => 2p - 1 (entre -1 y 1) por el desvio esperado a `horizonte` ruedas.
        return math.tanh(score / 2) * vol * math.sqrt(self.config.horizonte)

    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        resultados = super().predict_panel(self._preparar(dfs))
        umbral = self.metrics.get("umbral_senal", 0.0) if self.config.cobertura_senal else 0.0
        for r in resultados.values():
            log_return = float(np.log(r["predicted_close"] / r["last_close"]))
            r["signal"] = senal(log_return, umbral)
        return resultados
