"""Métricas do alerta de insatisfação e seus intervalos, em NumPy.

Tudo aqui parte de dois vetores: `y` (1 = reclamação, 0 = não) e `escore` (quanto maior, mais
o método acha que é reclamação). O bootstrap reamostra avaliações com reposição.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy import optimize, stats


def auc(y: np.ndarray, escore: np.ndarray) -> float:
    """Área sob a curva ROC: a chance de uma reclamação sorteada ter escore maior que uma não reclamação
    sorteada (empate conta meio). Calculada pelos postos, como na estatística de Mann-Whitney."""
    return auc_dos_postos(y, stats.rankdata(escore))  # empates recebem o posto médio


def auc_dos_postos(y: np.ndarray, postos: np.ndarray) -> float:
    """AUC a partir dos postos já calculados do escore (útil para avaliar o mesmo escore contra vários rótulos)."""
    y = np.asarray(y, dtype=bool)
    positivos, negativos = int(y.sum()), int((~y).sum())
    if positivos == 0 or negativos == 0:
        return float("nan")
    return float((postos[y].sum() - positivos * (positivos + 1) / 2) / (positivos * negativos))


def captura_no_topo(y: np.ndarray, escore: np.ndarray, fracao: float) -> tuple[float, float]:
    """Se o time lê só a `fracao` de avaliações com maior escore: (revocação, precisão).

    Revocação = parcela das reclamações que caem na pilha lida; precisão = parcela da pilha que é
    reclamação. Quando há empate de escore na borda da pilha (comum no léxico, em que muitos textos
    têm escore zero), entra o valor esperado de sortear os empatados para completar a pilha.
    """
    revocacao, precisao = capturas(y, escore, np.array([fracao]))
    return float(revocacao[0]), float(precisao[0])


def capturas(y: np.ndarray, escore: np.ndarray, fracoes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Revocação e precisão para várias parcelas lidas de uma vez (ordena uma vez só)."""
    y = np.asarray(y, dtype=float)
    negativo = -np.asarray(escore, dtype=float)
    ordem = np.argsort(negativo, kind="stable")
    return _capturas_ordenadas(negativo[ordem], y[ordem], fracoes)


def _capturas_ordenadas(negativo: np.ndarray, y: np.ndarray, fracoes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """O mesmo que `capturas`, para avaliações já em ordem decrescente de escore (`negativo` = -escore, crescente)."""
    n, total = len(y), float(y.sum())
    acumulado = np.concatenate([[0.0], np.cumsum(y)])
    lidos = np.clip(np.asarray(fracoes, dtype=float), 0.0, 1.0) * n
    borda = np.minimum(np.floor(lidos).astype(int), n - 1)  # primeira avaliação que pode ficar de fora
    corte = negativo[borda]
    acima = np.searchsorted(negativo, corte, side="left")  # quantas têm escore estritamente maior que o da borda
    ate = np.searchsorted(negativo, corte, side="right")  # ... e quantas têm escore maior ou igual
    taxa_dos_empatados = (acumulado[ate] - acumulado[acima]) / (ate - acima)
    capturadas = acumulado[acima] + taxa_dos_empatados * (lidos - acima)
    with np.errstate(divide="ignore", invalid="ignore"):
        revocacao = capturadas / total if total > 0 else np.full(len(lidos), np.nan)
        precisao = np.where(lidos > 0, capturadas / lidos, np.nan)
    return revocacao, precisao


def auc_e_capturas(y: np.ndarray, escore: np.ndarray, fracoes: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(AUC, revocação e precisão para cada parcela lida) com uma única ordenação.

    Dá o mesmo resultado de `auc` e `capturas` chamadas em separado; existe porque, dentro do
    bootstrap, ordenar uma vez em vez de três corta o tempo pela metade. A AUC sai dos postos
    médios: empatados dividem o posto, como em `auc`.
    """
    y = np.asarray(y, dtype=float)
    escore = np.asarray(escore, dtype=float)
    ordem = np.argsort(escore, kind="stable")
    crescente, y_crescente = escore[ordem], y[ordem]
    inicio = np.concatenate([[0], np.flatnonzero(np.diff(crescente)) + 1])  # onde começa cada grupo de escores iguais
    fim = np.concatenate([inicio[1:], [len(y)]])
    posto_medio = (inicio + fim + 1) / 2.0  # postos de 1 a n; o grupo ocupa os postos inicio+1 ... fim
    positivos, negativos = y.sum(), len(y) - y.sum()
    if positivos == 0 or negativos == 0:
        area = float("nan")
    else:
        soma_dos_postos = float(np.sum(np.add.reduceat(y_crescente, inicio) * posto_medio))
        area = (soma_dos_postos - positivos * (positivos + 1) / 2) / (positivos * negativos)
    revocacao, precisao = _capturas_ordenadas(-crescente[::-1], y_crescente[::-1], fracoes)
    return float(area), revocacao, precisao


def curva_de_captura(y: np.ndarray, escore: np.ndarray, fracoes: np.ndarray) -> np.ndarray:
    """Revocação para cada parcela lida (a curva de ganho acumulado)."""
    return capturas(y, escore, fracoes)[0]


GRADE_DE_LEITURA = np.linspace(0.0, 1.0, 1001)


def leitura_para_capturar(y: np.ndarray, escore: np.ndarray, alvo: float) -> float:
    """Menor parcela das avaliações (em passos de 0,1%) que é preciso ler para pegar a parcela `alvo` das reclamações."""
    revocacao = curva_de_captura(y, escore, GRADE_DE_LEITURA)
    return float(GRADE_DE_LEITURA[np.argmax(revocacao >= alvo - 1e-12)])


def precisao_e_revocacao(y: np.ndarray, previsto: np.ndarray) -> tuple[float, float]:
    """Para uma regra de sim ou não (como "nota até 2"): (precisão, revocação)."""
    y, previsto = np.asarray(y, dtype=bool), np.asarray(previsto, dtype=bool)
    acertos = float((y & previsto).sum())
    precisao = acertos / previsto.sum() if previsto.sum() else float("nan")
    revocacao = acertos / y.sum() if y.sum() else float("nan")
    return precisao, revocacao


# ---------------------------------------------------------------------------------------------
# Incerteza
# ---------------------------------------------------------------------------------------------


def bootstrap(
    estatistica: Callable[..., float | np.ndarray],
    *vetores: np.ndarray,
    repeticoes: int = 1000,
    semente: int = 42,
    nivel: float = 0.95,
) -> tuple:
    """Intervalo percentil: reamostra as avaliações com reposição e recalcula a estatística.

    Todos os `vetores` são reamostrados com os mesmos índices, então a estatística pode comparar
    dois métodos nas mesmas avaliações (bootstrap pareado): basta ela devolver a diferença.
    Se a estatística devolver um vetor (várias métricas de uma vez), os limites também são vetores.
    """
    return intervalo(reamostragens(estatistica, *vetores, repeticoes=repeticoes, semente=semente), nivel)


def reamostragens(estatistica: Callable[..., float | np.ndarray], *vetores: np.ndarray, repeticoes: int = 1000, semente: int = 42) -> np.ndarray:
    """O valor da estatística em cada reamostragem (uma linha por repetição).

    Guardar as reamostragens permite tirar, do mesmo sorteio, o intervalo de cada método e o da
    diferença entre dois métodos.
    """
    gerador = np.random.default_rng(semente)
    n = len(vetores[0])
    valores = []
    for _ in range(repeticoes):
        indices = gerador.integers(0, n, size=n)
        valores.append(estatistica(*(v[indices] for v in vetores)))
    return np.asarray(valores, dtype=float)


def intervalo(valores: np.ndarray, nivel: float = 0.95) -> tuple:
    """Limites percentis das reamostragens (por coluna, se houver mais de uma métrica)."""
    cauda = (1.0 - nivel) / 2.0
    with warnings.catch_warnings():  # métrica indefinida em todas as reamostragens (grupo de uma classe só) fica NaN, sem aviso
        warnings.simplefilter("ignore", RuntimeWarning)
        inferior = np.nanquantile(valores, cauda, axis=0)
        superior = np.nanquantile(valores, 1.0 - cauda, axis=0)
    if valores.ndim == 1:
        return float(inferior), float(superior)
    return inferior, superior


def intervalo_de_wilson(sucessos: int, total: int, nivel: float = 0.95) -> tuple[float, float]:
    """Intervalo de Wilson para uma proporção (funciona bem perto de 0 e de 1 e com amostra pequena)."""
    if total == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1.0 - (1.0 - nivel) / 2.0)
    p = sucessos / total
    centro = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    meia = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return float(max(0.0, centro - meia)), float(min(1.0, centro + meia))


# ---------------------------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------------------------


def brier(y: np.ndarray, probabilidade: np.ndarray) -> float:
    """Erro quadrático médio da probabilidade (0 é perfeito)."""
    return float(np.mean((np.asarray(probabilidade) - np.asarray(y, dtype=float)) ** 2))


def curva_de_calibracao(y: np.ndarray, probabilidade: np.ndarray, faixas: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide as avaliações em faixas de mesmo tamanho pela probabilidade prevista e devolve, por faixa,
    (probabilidade média prevista, frequência observada de reclamação, número de avaliações)."""
    y, probabilidade = np.asarray(y, dtype=float), np.asarray(probabilidade, dtype=float)
    ordem = np.argsort(probabilidade, kind="stable")
    grupos = np.array_split(ordem, faixas)
    prevista = np.array([probabilidade[g].mean() for g in grupos])
    observada = np.array([y[g].mean() for g in grupos])
    tamanho = np.array([len(g) for g in grupos])
    return prevista, observada, tamanho


def erro_de_calibracao(y: np.ndarray, probabilidade: np.ndarray, faixas: int = 10) -> float:
    """ECE: distância média, ponderada pelo tamanho da faixa, entre o previsto e o observado."""
    prevista, observada, tamanho = curva_de_calibracao(y, probabilidade, faixas)
    return float(np.sum(tamanho * np.abs(prevista - observada)) / tamanho.sum())


def ajustar_platt(logito: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Calibração de Platt: acha (a, b) tais que sigmoide(a * logito + b) tenha a menor perda logarítmica.

    É uma regressão logística de uma variável só, ajustada no período de validação: corrige um
    escore sistematicamente confiante demais (a < 1) ou deslocado (b diferente de 0).
    """
    logito, y = np.asarray(logito, dtype=float), np.asarray(y, dtype=float)

    def perda(parametros: np.ndarray) -> float:
        z = parametros[0] * logito + parametros[1]
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def gradiente(parametros: np.ndarray) -> np.ndarray:
        erro = sigmoide(parametros[0] * logito + parametros[1]) - y
        return np.array([np.mean(erro * logito), np.mean(erro)])

    resultado = optimize.minimize(perda, x0=np.array([1.0, 0.0]), jac=gradiente, method="BFGS")
    return float(resultado.x[0]), float(resultado.x[1])


def sigmoide(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
