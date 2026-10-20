"""Transformer de panel: mira todas las acciones del dia juntas (fase 4 del plan).

Parte de las mismas features y target que el Ridge / GBM de panel
(``src.panel_features``: normalizadas por vol y rankeadas por dia) y agrega lo
que un modelo lineal no puede ver:

1. **Encoder temporal compartido**: cada accion entra con la ventana de sus
   ultimas ``ventana`` ruedas de features ``[W, F]`` (mas un canal ``presente``
   que distingue un 0 imputado de un dato). Proyeccion lineal + posicion
   aprendida + token CLS + ``capas_temporales`` capas de encoder => ``h_i``.
2. **Embeddings** de sector (y opcionalmente de ticker: riesgo de memorizar)
   sumados a ``h_i``. El ticker se indexa por su nombre en el universo, no por
   su posicion en el panel.
3. **Token macro** (opcional): el contexto macro del dia (``contexto_macro``,
   con los retrasos de publicacion) proyectado a ``d_model`` y antepuesto a las
   acciones como un elemento mas.
4. **Atencion entre acciones del mismo dia** (``capas_cruzadas`` capas) con
   ``src_key_padding_mask`` para los tickers ausentes ese dia.
5. **Cabeza lineal** por accion => score en unidades del target (vol a
   ``horizonte`` ruedas, sin la media del dia).

Cada muestra es un dia (todas sus acciones); un batch es un conjunto de dias.
Perdida (``perdida``): ``"huber"`` sobre el target, ``"ic"`` (menos la
correlacion de Pearson por dia entre score y target: IC diferenciable) o
``"combinada"`` (``huber + lambda_ic * (-ic)``). Cada dia pesa segun su
antiguedad (``pesos_temporales``). Early stopping en validacion, con
``particion`` (gap = horizonte), segun ``seleccion``: ``"ic"`` (Spearman por
dia: ordenar bien las acciones), ``"direccion"`` (``acierto_balanceado`` del
signo del score contra el del target: acertar si sube o baja; tiene sentido
con ``target="absoluto"``, que no le resta la media del dia) o ``"perdida"``
(la misma perdida del entrenamiento: con ~75 dias de validacion, las otras dos
son tan ruidosas que la epoca elegida es casi al azar). ``n_semillas`` redes
se promedian.

Como los demas modelos de panel, predice con ``predict_panel(dfs)`` y recibe el
CCL / la macro por ``proveedor(hasta) -> (ccl, macro)``. ``save`` / ``load``
persisten redes, config y normalizadores (no el proveedor: se pasa al cargar).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.errors import ApiMlError, NotEnoughDataError, StaleArtifactError
from src.panel_datos import SECTORES, universo
from src.panel_features import (
    NOMBRES,
    FeaturesPanel,
    Normalizador,
    PreprocesoConfig,
    particion,
    pesos_temporales,
)
from src.panel_gbm import RUEDAS_PREDICCION, Proveedor, features_y_contexto
from src.trend_common import acierto_balanceado, derive_trend_output, ic_por_fecha

PERDIDAS = ("huber", "ic", "combinada")
SELECCIONES = ("ic", "direccion", "perdida")
SECTORES_LISTA = [*sorted(set(SECTORES.values())), "otro"]
MIN_DIAS = 100


@dataclass(frozen=True)
class PanelTransformerConfig:
    # Datos (como PanelGBMConfig)
    universo: str = "panel"
    moneda: str = "usd"
    target: str = "sin_media"
    macro: bool = False  # token macro del dia
    vida_media_anios: float | None = 2.0
    horizonte: int = 5
    zona_neutral: float = 0.0  # ver PreprocesoConfig
    # Red
    ventana: int = 40
    d_model: int = 32
    nhead: int = 4
    dim_feedforward: int = 64
    capas_temporales: int = 1
    capas_cruzadas: int = 1
    dropout: float = 0.2
    emb_ticker: bool = False
    emb_sector: bool = True
    # Perdida
    perdida: str = "combinada"
    lambda_ic: float = 1.0
    huber_delta: float = 1.0
    # Metrica de early stopping en validacion (ver docstring del modulo).
    seleccion: str = "ic"
    # Entrenamiento
    epocas: int = 60
    batch_dias: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epocas: int = 3
    paciencia: int = 10
    val_fraction: float = 0.15
    min_tickers: int = 5  # tickers con target para que un dia sea muestra
    n_semillas: int = 1
    seed: int = 0

    def perdidas_validas(self) -> tuple[str, ...]:
        """Perdidas que acepta la config (una subclase puede sumar las suyas)."""
        return PERDIDAS

    def __post_init__(self) -> None:
        universo(self.universo)
        if self.perdida not in self.perdidas_validas():
            raise ValueError(f"perdida {self.perdida!r} desconocida ({self.perdidas_validas()})")
        if self.seleccion not in SELECCIONES:
            raise ValueError(f"seleccion {self.seleccion!r} desconocida ({SELECCIONES})")
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) debe ser divisible por nhead ({self.nhead})"
            )
        if self.ventana < 1 or self.ventana > RUEDAS_PREDICCION - 260:
            raise ValueError(f"ventana debe estar entre 1 y {RUEDAS_PREDICCION - 260}")
        if self.capas_temporales < 1 or self.capas_cruzadas < 0:
            raise ValueError("hace falta al menos una capa temporal (y capas_cruzadas >= 0)")
        if self.n_semillas < 1:
            raise ValueError("n_semillas debe ser al menos 1")

    def preproceso(self) -> PreprocesoConfig:
        return PreprocesoConfig(
            horizonte=self.horizonte, target=self.target, zona_neutral=self.zona_neutral
        )


# --------------------------------------------------------------------------- #
# Red
# --------------------------------------------------------------------------- #
def _encoder(cfg: PanelTransformerConfig, capas: int) -> nn.TransformerEncoder:
    capa = nn.TransformerEncoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )
    return nn.TransformerEncoder(capa, num_layers=capas, enable_nested_tensor=False)


class _RedPanel(nn.Module):
    def __init__(self, cfg: PanelTransformerConfig, n_features: int, n_macro: int, n_tickers: int):
        super().__init__()
        d = cfg.d_model
        self.proyeccion = nn.Linear(n_features, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.posicion = nn.Parameter(torch.zeros(1, cfg.ventana + 1, d))
        for p in (self.cls, self.posicion):
            nn.init.trunc_normal_(p, std=0.02)
        self.temporal = _encoder(cfg, cfg.capas_temporales)
        self.norm_temporal = nn.LayerNorm(d)
        self.emb_ticker = nn.Embedding(n_tickers, d) if cfg.emb_ticker else None
        self.emb_sector = nn.Embedding(len(SECTORES_LISTA), d) if cfg.emb_sector else None
        for emb in (self.emb_ticker, self.emb_sector):
            if emb is not None:
                nn.init.normal_(emb.weight, std=0.02)
        self.macro = nn.Linear(n_macro, d) if n_macro else None
        self.tipo_macro = nn.Parameter(torch.zeros(1, 1, d)) if n_macro else None
        # Sin capas cruzadas (ablacion) cada accion se predice sola, como un modelo por ticker.
        self.cruzada = _encoder(cfg, cfg.capas_cruzadas) if cfg.capas_cruzadas else None
        self.norm = nn.LayerNorm(d)
        self.cabeza = nn.Linear(d, 1)

    def forward(
        self,
        x: torch.Tensor,
        presentes: torch.Tensor,
        tickers: torch.Tensor,
        sectores: torch.Tensor,
        macro: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x [B, N, W, F]``, ``presentes [B, N]`` (bool), ``tickers`` y
        ``sectores [N]`` (indices), ``macro [B, M]`` => score ``[B, N]``."""
        b, n, w, f = x.shape
        h = self.proyeccion(x.reshape(b * n, w, f))
        h = torch.cat([self.cls.expand(b * n, -1, -1), h], dim=1) + self.posicion[:, : w + 1]
        h = self.norm_temporal(self.temporal(h)[:, 0]).reshape(b, n, -1)
        if self.emb_ticker is not None:
            h = h + self.emb_ticker(tickers)[None]
        if self.emb_sector is not None:
            h = h + self.emb_sector(sectores)[None]
        ausentes = ~presentes
        if self.macro is not None:
            token = self.macro(macro)[:, None] + self.tipo_macro
            h = torch.cat([token, h], dim=1)
            ausentes = torch.cat([torch.zeros_like(ausentes[:, :1]), ausentes], dim=1)
        if self.cruzada is not None:
            h = self.cruzada(h, src_key_padding_mask=ausentes)
        if self.macro is not None:
            h = h[:, 1:]
        return self.cabeza(self.norm(h)).squeeze(-1)


def _ventanas(x: torch.Tensor, dias: torch.Tensor, ventana: int) -> torch.Tensor:
    """De ``x [T, N, F]`` a las ventanas ``[B, N, W, F]`` que terminan en ``dias``."""
    idx = dias[:, None] + torch.arange(-ventana + 1, 1, device=x.device)[None]
    return x[idx].permute(0, 2, 1, 3)


def ic_pearson_por_dia(score: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Correlacion de Pearson por fila entre ``score`` e ``y`` en las celdas de
    ``m`` (``[B, N]`` => ``[B]``). Diferenciable: es el IC que se optimiza."""
    mf = m.float()
    n = mf.sum(1).clamp(min=1)
    s = (score - (score * mf).sum(1, keepdim=True) / n[:, None]) * mf
    t = (y - (y * mf).sum(1, keepdim=True) / n[:, None]) * mf
    return (s * t).sum(1) / torch.sqrt((s * s).sum(1) * (t * t).sum(1) + 1e-8)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
class PanelTransformerModel:
    # Lo que identifica al artefacto; una subclase con otra config los redefine.
    ARQUITECTURA = "panel_transformer"
    CONFIG = PanelTransformerConfig

    def __init__(
        self,
        config: PanelTransformerConfig | None = None,
        proveedor: Proveedor | None = None,
        dispositivo: str = "cpu",
    ):
        self.config = config or self.CONFIG()
        self.proveedor = proveedor
        # Dispositivo de entrenamiento; las redes vuelven a CPU al terminar.
        self.dispositivo = dispositivo
        self.universo = universo(self.config.universo)
        self.metrics: dict = {}
        self.trained_at: str | None = None
        self._redes: list[_RedPanel] = []
        self._normalizador: Normalizador | None = None
        self._normalizador_macro: Normalizador | None = None
        self._columnas_macro: list[str] = []

    @property
    def version(self) -> str:
        return f"panel-transformer-{self.config.moneda}"

    # ------------------------------------------------------------------ datos
    def _tensores(
        self, fp: FeaturesPanel, ctx: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Features normalizadas con canal ``presente`` ``[T, N, F+1]``, indices
        de ticker y sector ``[N]`` y macro normalizada ``[T, M]`` (o None)."""
        x = np.where(fp.mascara[..., None], self._normalizador.transform(fp.valores), 0.0)
        x = np.concatenate([x, fp.mascara[..., None].astype(float)], axis=-1)
        tickers = np.array([self.universo.index(t) for t in fp.tickers])
        sectores = np.array([SECTORES_LISTA.index(s) for s in fp.sectores])
        macro = None
        if self.config.macro:
            m = ctx.reindex(columns=self._columnas_macro).to_numpy(dtype=float)
            macro = np.nan_to_num(self._normalizador_macro.transform(m))
        return x.astype(np.float32), tickers, sectores, macro

    def _dias_muestra(self, fp: FeaturesPanel) -> np.ndarray:
        """Dias con ventana completa y al menos ``min_tickers`` targets."""
        n = fp.mascara_target.sum(axis=1)
        dias = np.nonzero(n >= self.config.min_tickers)[0]
        return dias[dias >= self.config.ventana - 1]

    # ---------------------------------------------------------- entrenamiento
    def fit(self, histories: dict[str, pd.DataFrame]) -> dict:
        cfg = self.config
        fp, ctx = features_y_contexto(histories, cfg, self.proveedor)
        dias = self._dias_muestra(fp)
        if len(dias) < MIN_DIAS:
            raise NotEnoughDataError(f"muy pocos dias para entrenar ({len(dias)})")
        tr, va = particion(fp.fechas[dias], cfg.val_fraction, gap=cfg.horizonte)
        dias_tr, dias_va = dias[tr], dias[va]
        # Normalizadores solo con las celdas / fechas de entrenamiento (hasta el ultimo
        # dia de train: las ventanas miran hacia atras).
        hasta = dias_tr[-1] + 1
        self._normalizador = Normalizador().fit(fp.valores[:hasta], fp.mascara[:hasta])
        if cfg.macro:
            self._columnas_macro = list(ctx.columns)
            self._normalizador_macro = Normalizador().fit(ctx.to_numpy(dtype=float)[:hasta])
        x, tickers, sectores, macro = self._tensores(fp, ctx)

        disp = torch.device(self.dispositivo)
        macro_t = None if macro is None else torch.tensor(macro, dtype=torch.float32, device=disp)
        datos = {
            "x": torch.tensor(x, device=disp),
            "presentes": torch.tensor(fp.mascara, device=disp),
            "tickers": torch.tensor(tickers, device=disp),
            "sectores": torch.tensor(sectores, device=disp),
            "macro": macro_t,
            "y": torch.tensor(np.nan_to_num(fp.target), dtype=torch.float32, device=disp),
            "m": torch.tensor(fp.mascara_target, device=disp),
            "w": torch.tensor(
                pesos_temporales(fp.fechas, cfg.vida_media_anios), dtype=torch.float32, device=disp
            ),
        }
        self._redes, por_red = [], []
        for k in range(cfg.n_semillas):
            red, info = self._entrenar_red(cfg.seed + k, datos, dias_tr, dias_va, x.shape[-1])
            self._redes.append(red)
            por_red.append(info)

        self.trained_at = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.metrics = {
            "dias_train": int(len(dias_tr)),
            "dias_val": int(len(dias_va)),
            f"val_{cfg.seleccion}": round(float(np.mean([i["val"] for i in por_red])), 4),
            "epoca_mejor": [i["epoca_mejor"] for i in por_red],
        }
        return self.metrics

    def _forward(self, red: _RedPanel, datos: dict, dias: torch.Tensor) -> torch.Tensor:
        macro = None if datos["macro"] is None else datos["macro"][dias]
        return red(
            _ventanas(datos["x"], dias, self.config.ventana),
            datos["presentes"][dias],
            datos["tickers"],
            datos["sectores"],
            macro,
        )

    def _perdida(self, score: torch.Tensor, datos: dict, dias: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        y, m, w = datos["y"][dias], datos["m"][dias], datos["w"][dias]
        total = score.new_zeros(())
        if cfg.perdida in ("huber", "combinada"):
            h = F.huber_loss(score, y, reduction="none", delta=cfg.huber_delta) * m
            por_dia = h.sum(1) / m.sum(1).clamp(min=1)
            total = total + (por_dia * w).sum() / w.sum()
        if cfg.perdida in ("ic", "combinada"):
            ic = ic_pearson_por_dia(score, y, m)
            peso = 1.0 if cfg.perdida == "ic" else cfg.lambda_ic
            total = total - peso * (ic * w).sum() / w.sum()
        return total

    def _metrica_validacion(self, red: _RedPanel, datos: dict, dias: torch.Tensor) -> float:
        """Metrica de early stopping en ``dias`` (mayor es mejor): IC de Spearman
        medio por dia, acierto balanceado de la direccion o menos la perdida de
        entrenamiento (la mas estable con pocos dias de validacion), segun
        ``seleccion``."""
        red.eval()
        with torch.no_grad():
            scores = torch.cat(
                [self._forward(red, datos, dias[i : i + 64]) for i in range(0, len(dias), 64)]
            )
        if self.config.seleccion == "perdida":
            with torch.no_grad():
                return -float(self._perdida(scores, datos, dias))
        m = datos["m"][dias].cpu().numpy()
        score, y = scores.cpu().numpy()[m], datos["y"][dias].cpu().numpy()[m]
        if self.config.seleccion == "direccion":
            return acierto_balanceado(score, y)
        fila, _ = np.nonzero(m)
        return ic_por_fecha(score, y, fila)

    def _entrenar_red(
        self,
        seed: int,
        datos: dict,
        dias_tr: np.ndarray,
        dias_va: np.ndarray,
        n_features: int,
    ) -> tuple[_RedPanel, dict]:
        cfg = self.config
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        disp = datos["x"].device
        n_macro = 0 if datos["macro"] is None else datos["macro"].shape[1]
        red = _RedPanel(cfg, n_features, n_macro, len(self.universo)).to(disp)
        optimizador = torch.optim.AdamW(
            red.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        # Warmup lineal y despues decaimiento coseno, como en src/transformer.py.
        def lr_lambda(epoca: int) -> float:
            if cfg.warmup_epocas > 0 and epoca < cfg.warmup_epocas:
                return (epoca + 1) / cfg.warmup_epocas
            progreso = (epoca - cfg.warmup_epocas) / max(1, cfg.epocas - cfg.warmup_epocas)
            return 0.5 * (1 + math.cos(math.pi * min(progreso, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizador, lr_lambda)
        va_t = torch.tensor(dias_va, device=disp)
        mejor, mejor_estado, epoca_mejor, sin_mejora = -math.inf, None, 0, 0
        for epoca in range(cfg.epocas):
            red.train()
            orden = rng.permutation(dias_tr)
            perdida_epoca = 0.0
            for i in range(0, len(orden), cfg.batch_dias):
                lote = torch.tensor(orden[i : i + cfg.batch_dias], device=disp)
                optimizador.zero_grad()
                perdida = self._perdida(self._forward(red, datos, lote), datos, lote)
                perdida.backward()
                nn.utils.clip_grad_norm_(red.parameters(), max_norm=1.0)
                optimizador.step()
                perdida_epoca += perdida.item() * len(lote)
            scheduler.step()

            val = self._metrica_validacion(red, datos, va_t)
            val = val if np.isfinite(val) else -1.0
            if val > mejor + 1e-5:
                mejor, epoca_mejor, sin_mejora = val, epoca + 1, 0
                mejor_estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
            else:
                sin_mejora += 1
            if (epoca + 1) % 10 == 0 or epoca == 0:
                print(
                    f"  epoca {epoca + 1:>3}/{cfg.epocas}  "
                    f"perdida={perdida_epoca / len(orden):.5f}  val_{cfg.seleccion}={val:.4f}"
                )
            if sin_mejora >= cfg.paciencia:
                print(f"  early stopping en epoca {epoca + 1} (mejor val={mejor:.4f})")
                break

        red.load_state_dict(mejor_estado)
        red.eval()
        return red.cpu(), {"val": mejor, "epoca_mejor": epoca_mejor}

    # -------------------------------------------------------------- prediccion
    def predict_panel(self, dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Prediccion de la ultima rueda para cada ticker de ``dfs`` (todos al
        mismo dia). Los que no tienen dato valido ese dia quedan afuera."""
        if not self._redes:
            raise NotEnoughDataError("el modelo de panel no esta entrenado")
        recortes = {s: df.iloc[-RUEDAS_PREDICCION:] for s, df in dfs.items() if not df.empty}
        if not recortes:
            return {}
        try:
            fp, ctx = features_y_contexto(recortes, self.config, self.proveedor)
        except ValueError:  # ningun ticker del universo
            return {}
        ultimo = len(fp.fechas) - 1
        presentes = fp.mascara[ultimo]
        if ultimo < self.config.ventana - 1 or not presentes.any():
            return {}
        x, tickers, sectores, macro = self._tensores(fp, ctx)
        datos = {
            "x": torch.tensor(x),
            "presentes": torch.tensor(fp.mascara),
            "tickers": torch.tensor(tickers),
            "sectores": torch.tensor(sectores),
            "macro": None if macro is None else torch.tensor(macro, dtype=torch.float32),
        }
        dia = torch.tensor([ultimo])
        with torch.no_grad():
            scores = np.mean(
                [self._forward(red, datos, dia)[0].numpy() for red in self._redes], axis=0
            )
        scores = self._ajustar_scores(scores, fp, ultimo)
        resultados = {}
        for n in np.nonzero(presentes)[0]:
            symbol = fp.tickers[n]
            log_return = self._a_log_return(float(scores[n]), float(fp.vol[ultimo, n]))
            resultados[symbol] = derive_trend_output(
                recortes[symbol], log_return, self.config.horizonte
            )
        return resultados

    def _ajustar_scores(self, scores: np.ndarray, fp: FeaturesPanel, dia: int) -> np.ndarray:
        """Post-proceso de los scores ``[N]`` del dia ``dia`` antes de pasarlos a
        retorno (una subclase puede, p. ej., centrarlos o ensamblarlos)."""
        return scores

    def _a_log_return(self, score: float, vol: float) -> float:
        """El score esta en unidades de vol a ``horizonte`` ruedas: se lleva a retorno log."""
        return score * vol * float(np.sqrt(self.config.horizonte))

    def predict_df(self, df: pd.DataFrame) -> dict:
        raise ApiMlError(
            "el Transformer de panel predice con todos los tickers del dia (predict_panel)"
        )

    # ----------------------------------------------------------- persistencia
    def save(self, path: str | Path) -> None:
        if not self._redes:
            raise NotEnoughDataError("no hay un modelo de panel entrenado para guardar")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        macro = self._normalizador_macro
        torch.save(
            {
                "architecture": self.ARQUITECTURA,
                "config": asdict(self.config),
                "state_dicts": [red.state_dict() for red in self._redes],
                "normalizador": (self._normalizador.media, self._normalizador.desvio),
                "normalizador_macro": None if macro is None else (macro.media, macro.desvio),
                "columnas_macro": self._columnas_macro,
                # Para detectar artefactos viejos: las features y el universo (indice
                # del embedding de ticker) tienen que coincidir con los actuales.
                "feature_names": list(NOMBRES),
                "universo": self.universo,
                "trained_at": self.trained_at,
                "metrics": self.metrics,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, proveedor: Proveedor | None = None, dispositivo: str = "cpu"
    ) -> PanelTransformerModel:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        if blob.get("architecture") != cls.ARQUITECTURA:
            raise ApiMlError(f"{path} no es un artefacto de {cls.ARQUITECTURA}")
        modelo = cls(cls.CONFIG(**blob["config"]), proveedor, dispositivo)
        for campo, actual in (("feature_names", list(NOMBRES)), ("universo", modelo.universo)):
            if list(blob[campo]) != actual:
                raise StaleArtifactError(
                    f"artefacto entrenado con otro {campo} (guardado={list(blob[campo])}, "
                    f"actual={actual}); hay que reentrenar"
                )
        modelo._normalizador = Normalizador(*blob["normalizador"])
        if blob["normalizador_macro"] is not None:
            modelo._normalizador_macro = Normalizador(*blob["normalizador_macro"])
        modelo._columnas_macro = list(blob["columnas_macro"])
        modelo.trained_at = blob["trained_at"]
        modelo.metrics = blob["metrics"]
        # +1 por el canal "presente" que agrega _tensores.
        n_features = len(modelo._normalizador.media) + 1
        n_macro = len(modelo._columnas_macro) if modelo.config.macro else 0
        for state in blob["state_dicts"]:
            red = _RedPanel(modelo.config, n_features, n_macro, len(modelo.universo))
            red.load_state_dict(state)
            red.eval()
            modelo._redes.append(red)
        return modelo

    def resumen(self) -> dict:
        return {"version": self.version, "config": asdict(self.config), "metrics": self.metrics}
