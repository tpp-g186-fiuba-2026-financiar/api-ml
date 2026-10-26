"""Nombres en castellano simple de las variables de los modelos, para mostrarle
al usuario que empujo una prediccion (ver ``XGBTrendModel.explain``).

Las ``r_<variable>`` (rango frente al resto de las acciones ese dia, ver
``src.xgb_trend``) reusan el nombre de la variable base con un sufijo.
"""

# ruff: noqa: E501 -- son textos para el usuario, no se cortan

from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    # Tecnicas de la accion (src.macro_trend.technical_features)
    "t_ret1": "Variación del último día",
    "t_ret3": "Variación de los últimos 3 días",
    "t_ret5": "Variación de la última semana",
    "t_ret10": "Variación de las últimas 2 semanas",
    "t_ret20": "Variación del último mes",
    "t_ret60": "Variación de los últimos 3 meses",
    "t_ret120": "Variación de los últimos 6 meses",
    "t_ret250": "Variación del último año",
    "t_down_streak": "Días seguidos en baja",
    "t_up_streak": "Días seguidos en alza",
    "t_dd252": "Distancia al máximo del año",
    "tz126": "Distancia a su tendencia de 6 meses",
    "tz252": "Distancia a su tendencia del año",
    "tslope126": "Tendencia de 6 meses (por año)",
    "tslope252": "Tendencia del año (por año)",
    "tr2_126": "Qué tan marcada es la tendencia de 6 meses",
    "tr2_252": "Qué tan marcada es la tendencia del año",
    "t_dd60": "Distancia al máximo de 3 meses",
    "t_dist_min60": "Distancia al mínimo de 3 meses",
    "t_new_low252": "Mínimo del año",
    "t_sma20": "Precio vs. promedio de 1 mes",
    "t_sma50": "Precio vs. promedio de 50 días",
    "t_sma200": "Precio vs. promedio de 200 días",
    "t_death_cross": "Promedio de 50 días debajo del de 200",
    "t_gap": "Salto de apertura",
    "t_gap_down": "Apertura con caída fuerte",
    "t_vol5": "Volatilidad de la última semana",
    "t_vol20": "Volatilidad del último mes",
    "t_vol60": "Volatilidad de los últimos 3 meses",
    "t_vol_ratio5_20": "Volatilidad reciente vs. del mes",
    "t_vol_ratio20_60": "Volatilidad del mes vs. del trimestre",
    "t_downdev": "Peso de las caídas en la volatilidad",
    "t_skew20": "Asimetría de las subas y bajas (último mes)",
    "t_kurt20": "Movimientos extremos (último mes)",
    "t_gk20": "Rango diario de precios (último mes)",
    "t_gk_ratio": "Rango diario reciente vs. del mes",
    "t_volz": "Volumen operado vs. lo habitual",
    "t_volz_x_ret": "Volumen inusual en la dirección del día",
    "t_vol_chg5": "Cambio del volumen en la semana",
    "t_obv_slope": "Presión compradora (volumen)",
    "t_amihud": "Iliquidez",
    "t_rsi14": "RSI (14 días)",
    "t_rsi5": "RSI (5 días)",
    "t_macd": "MACD",
    "t_stoch": "Estocástico (posición en el rango de 14 días)",
    "t_boll": "Posición en las bandas de Bollinger",
    # Contexto de mercado (src.context_trend.context_features)
    "m_ret5": "Mercado: variación de la semana",
    "m_ret20": "Mercado: variación del mes",
    "m_ret60": "Mercado: variación de 3 meses",
    "m_vol20": "Mercado: volatilidad del mes",
    "m_dd252": "Mercado: distancia al máximo del año",
    "m_breadth50": "Mercado: acciones sobre su promedio de 50 días",
    "m_rel5": "Rendimiento vs. el mercado (semana)",
    "m_rel20": "Rendimiento vs. el mercado (mes)",
    "m_rel60": "Rendimiento vs. el mercado (3 meses)",
    "m_beta60": "Sensibilidad al mercado (beta)",
    # Macro (src.macro_trend.macro_features)
    "a_ccl1": "Dólar CCL (último día)",
    "a_ccl5": "Dólar CCL (última semana)",
    "a_ccl20": "Dólar CCL (último mes)",
    "a_ccl60": "Dólar CCL (últimos 3 meses)",
    "a_brecha": "Brecha cambiaria",
    "a_brecha_chg20": "Brecha cambiaria (cambio del mes)",
    "a_brecha_z": "Brecha cambiaria vs. lo habitual",
    "a_mep_ccl": "Dólar MEP vs. CCL",
    "a_blue_gap": "Brecha del dólar blue",
    "a_of20": "Dólar oficial (último mes)",
    "a_rp": "Riesgo país",
    "a_rp_chg5": "Riesgo país (última semana)",
    "a_rp_chg20": "Riesgo país (último mes)",
    "a_rp_z": "Riesgo país vs. lo habitual",
    "a_res20": "Reservas del BCRA (último mes)",
    "a_res60": "Reservas del BCRA (últimos 3 meses)",
    "a_badlar": "Tasa BADLAR",
    "a_badlar_chg20": "Tasa BADLAR (cambio del mes)",
    "a_bm20": "Base monetaria (último mes)",
    "a_real_rate": "Tasa real",
}

RANK_SUFFIX = " (vs. otras acciones)"

# Que mide cada variable, en una o dos frases para alguien que no sabe finanzas.
FEATURE_HELP: dict[str, str] = {
    "t_ret1": "Cuánto subió o bajó el precio ayer.",
    "t_ret3": "Cuánto subió o bajó el precio en los últimos 3 días.",
    "t_ret5": "Cuánto subió o bajó el precio en la última semana.",
    "t_ret10": "Cuánto subió o bajó el precio en las últimas 2 semanas.",
    "t_ret20": "Cuánto subió o bajó el precio en el último mes.",
    "t_ret60": "Cuánto subió o bajó el precio en los últimos 3 meses.",
    "t_ret120": "Cuánto subió o bajó el precio en los últimos 6 meses.",
    "t_ret250": "Cuánto subió o bajó el precio en el último año.",
    "t_down_streak": "Cuántas ruedas seguidas viene cerrando en baja.",
    "t_up_streak": "Cuántas ruedas seguidas viene cerrando en alza.",
    "t_dd252": "Cuánto está por debajo del precio más alto del último año (0% = está en el máximo).",
    "tz126": "Cuántos desvíos está el precio por encima (+) o por debajo (−) de la recta de regresión de los últimos 6 meses.",
    "tz252": "Cuántos desvíos está el precio por encima (+) o por debajo (−) de la recta de regresión del último año.",
    "tslope126": "Cuánto sube o baja por año la recta de regresión de los últimos 6 meses.",
    "tslope252": "Cuánto sube o baja por año la recta de regresión del último año.",
    "tr2_126": "Qué parte del movimiento de los últimos 6 meses explica la recta (R², de 0 a 1).",
    "tr2_252": "Qué parte del movimiento del último año explica la recta (R², de 0 a 1).",
    "t_dd60": "Cuánto está por debajo del precio más alto de los últimos 3 meses (0% = está en el máximo).",
    "t_dist_min60": "Cuánto está por encima del precio más bajo de los últimos 3 meses.",
    "t_new_low252": "Si hoy marcó el precio más bajo del año.",
    "t_sma20": "Cuánto está el precio por encima o por debajo de su promedio del último mes.",
    "t_sma50": "Cuánto está el precio por encima o por debajo de su promedio de 50 ruedas.",
    "t_sma200": "Cuánto está el precio por encima o por debajo de su promedio de 200 ruedas (tendencia larga).",
    "t_death_cross": "Si el promedio de 50 ruedas está debajo del de 200: señal clásica de tendencia bajista.",
    "t_gap": "Cuánto abrió hoy por encima o por debajo del cierre anterior.",
    "t_gap_down": "Si hoy abrió más de 2% por debajo del cierre anterior.",
    "t_vol5": "Cuánto se movió el precio por día, en promedio, la última semana.",
    "t_vol20": "Cuánto se movió el precio por día, en promedio, el último mes.",
    "t_vol60": "Cuánto se movió el precio por día, en promedio, los últimos 3 meses.",
    "t_vol_ratio5_20": "Si esta semana está más movida (más de 1) o más tranquila (menos de 1) que el mes.",
    "t_vol_ratio20_60": "Si este mes está más movido (más de 1) o más tranquilo (menos de 1) que el trimestre.",
    "t_downdev": "Qué parte del movimiento del mes vino de días de caída (más alto = caídas más bruscas).",
    "t_skew20": "Si en el mes hubo más saltos grandes para arriba (positivo) o para abajo (negativo).",
    "t_kurt20": "Si en el mes hubo días con movimientos fuera de lo común (más alto = más extremos).",
    "t_gk20": "Distancia promedio entre el precio más alto y el más bajo de cada día, en el último mes.",
    "t_gk_ratio": "Si el rango diario de esta semana es mayor (más de 1) o menor que el del mes.",
    "t_volz": "Si hoy se operó mucho más (positivo) o menos (negativo) que lo habitual.",
    "t_volz_x_ret": "Volumen inusual combinado con la dirección del día: positivo = mucho volumen en un día de suba.",
    "t_vol_chg5": "Cuánto cambió el volumen operado en la última semana.",
    "t_obv_slope": "Si en el mes el volumen se concentró en días de suba (positivo) o de baja (negativo).",
    "t_amihud": "Cuánto mueve el precio cada peso operado, comparado con lo habitual (más alto = menos líquida).",
    "t_rsi14": "Indicador de 0 a 100: arriba de 70 suele leerse como 'subió mucho', abajo de 30 como 'cayó mucho'.",
    "t_rsi5": "Lo mismo que el RSI pero mirando solo la última semana.",
    "t_macd": "Diferencia entre la tendencia corta y la larga del precio: positivo = viene acelerando para arriba.",
    "t_stoch": "Dónde está el precio dentro del rango de las últimas 14 ruedas (0% = en el mínimo, 100% = en el máximo).",
    "t_boll": "Cuántos 'movimientos normales' se alejó el precio de su promedio del mes.",
    "m_ret5": "Cuánto subió o bajó el mercado (canasta de acciones líderes) en la semana.",
    "m_ret20": "Cuánto subió o bajó el mercado (canasta de acciones líderes) en el mes.",
    "m_ret60": "Cuánto subió o bajó el mercado (canasta de acciones líderes) en los últimos 3 meses.",
    "m_vol20": "Cuánto se movió el mercado por día, en promedio, el último mes.",
    "m_dd252": "Cuánto está el mercado por debajo de su máximo del año.",
    "m_breadth50": "Qué porcentaje de las acciones está por encima de su promedio de 50 ruedas.",
    "m_rel5": "Si a la acción le fue mejor (positivo) o peor que al mercado en la semana.",
    "m_rel20": "Si a la acción le fue mejor (positivo) o peor que al mercado en el mes.",
    "m_rel60": "Si a la acción le fue mejor (positivo) o peor que al mercado en los últimos 3 meses.",
    "m_beta60": "Cuánto se mueve la acción cuando se mueve el mercado: 1 = igual, más de 1 = amplifica.",
}

RANK_HELP = (
    " Acá no se mira el valor en sí sino el puesto: en qué lugar queda frente a"
    " las otras acciones ese mismo día."
)

# Como mostrar el valor: pct = proporcion (0.05 -> 5%), flag = si/no,
# days = ruedas, index = escala propia (RSI 0-100), rank = puesto 0-1.
_PCT = {
    *(f"t_ret{k}" for k in (1, 3, 5, 10, 20, 60, 120, 250)),
    *("t_dd252", "t_dd60", "t_dist_min60", "t_sma20", "t_sma50", "t_sma200", "t_gap"),
    *("t_vol5", "t_vol20", "t_vol60", "t_gk20", "t_vol_chg5", "t_stoch"),
    *("m_ret5", "m_ret20", "m_ret60", "m_vol20", "m_dd252", "m_breadth50"),
    *("m_rel5", "m_rel20", "m_rel60"),
    *("tslope126", "tslope252"),
}
_FLAGS = {"t_new_low252", "t_death_cross", "t_gap_down"}
_DAYS = {"t_down_streak", "t_up_streak"}
_INDEX = {"t_rsi14", "t_rsi5"}


def feature_label(name: str) -> str:
    """Nombre legible de una variable; el nombre tecnico si no hay traduccion."""
    if name.startswith("r_"):
        base = name[2:]
        if base in FEATURE_LABELS:
            return FEATURE_LABELS[base] + RANK_SUFFIX
    return FEATURE_LABELS.get(name, name)


def feature_help(name: str) -> str | None:
    """Que mide la variable, en palabras simples (None si no hay texto)."""
    if name.startswith("r_"):
        base = FEATURE_HELP.get(name[2:])
        return base + RANK_HELP if base else None
    return FEATURE_HELP.get(name)


def feature_unit(name: str) -> str:
    """Como formatear el valor: pct, flag, days, index, rank o num."""
    if name.startswith("r_"):
        return "rank"
    for unit, names in (("pct", _PCT), ("flag", _FLAGS), ("days", _DAYS), ("index", _INDEX)):
        if name in names:
            return unit
    return "num"
