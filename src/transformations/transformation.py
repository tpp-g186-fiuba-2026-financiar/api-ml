import pandas as pd


def add_moving_averages(prices: pd.Series, ma_as_array=[50, 200]) -> pd.DataFrame:
    """
    prices: serie de tiempo del precio de cierre (indexada por fecha)
    windows: lista de ventanas a calcular (mínimo 50 y 200 según el ticket)
    """
    df = pd.DataFrame({"price": prices})
    for ma in ma_as_array:
        df[f"sma_{ma}"] = prices.rolling(window=ma, min_periods=ma).mean()
    return df

def adjust_by_peso_value(stock_prices: pd.Series, fx_series: pd.Series) -> pd.Series:
    """
    stock_prices: serie de tiempo del precio de la acción (ARS como esta en data-collector)
    fx_series: serie de tiempo USD/ARS valor del peso en el mismo período
    devuelve: serie de tiempo con el precio ajustado
    """
    aligned_fx = fx_series.reindex(stock_prices.index).ffill()
    return stock_prices / aligned_fx