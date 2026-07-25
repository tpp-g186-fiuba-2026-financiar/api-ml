import numpy as np

from src.black_litterman.black_litterman import get_pi


def test_PI() -> None:
    m = [[1,2],[4,5]]
    n = [[7,8],[9,10]]
    alpha = 3
    resultado = get_pi(alpha, m, n)
    resultado_esperado = np.array([[75,84],[219,246]])
    assert(resultado, resultado_esperado)