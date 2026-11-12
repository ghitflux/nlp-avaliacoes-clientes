"""Classificadores de insatisfação: regressão logística sobre cada representação do texto.

Tudo o que é ajustado (vocabulário, IDF, pesos, regularização, calibração, projeção densa) vê só
o treino ou a validação; o teste entra apenas para medir.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

from nlp_avaliacoes import metricas

GRADE_DE_REGULARIZACAO = (0.3, 1.0, 3.0, 10.0)


def ajustar_logistica(matriz, y: np.ndarray, c: float = 1.0) -> LogisticRegression:
    """Regressão logística com penalidade L2 (C menor = mais regularização), com solver determinístico.

    Matriz esparsa (TF-IDF): liblinear, com tolerância de parada de 0,001 (o padrão é 0,0001), que
    reduz o tempo nas matrizes de caracteres e muda a AUC só na quarta casa decimal. Matriz densa
    (LSA, embeddings): L-BFGS, bem mais rápido nesse formato.
    """
    if sparse.issparse(matriz):
        return LogisticRegression(C=c, solver="liblinear", tol=1e-3, max_iter=500).fit(matriz, y)
    return LogisticRegression(C=c, solver="lbfgs", max_iter=2000).fit(matriz, y)


def escolher_regularizacao(
    treino, y_treino: np.ndarray, validacao, y_validacao: np.ndarray, grade: Sequence[float] = GRADE_DE_REGULARIZACAO
) -> tuple[LogisticRegression, pd.DataFrame]:
    """Ajusta um modelo por valor de C no treino e fica com o de maior AUC na validação.

    Devolve o modelo escolhido e a tabela com a AUC de validação de cada C (o teste não participa).
    """
    linhas, melhor, melhor_auc = [], None, -np.inf
    for c in grade:
        modelo = ajustar_logistica(treino, y_treino, c)
        valor = metricas.auc(y_validacao, modelo.decision_function(validacao))
        linhas.append({"C": c, "AUC na validação": valor})
        if valor > melhor_auc:
            melhor, melhor_auc = modelo, valor
    tabela = pd.DataFrame(linhas).set_index("C")
    tabela["escolhido"] = tabela["AUC na validação"] == melhor_auc
    return melhor, tabela


def projetar_lsa(treino: sparse.csr_matrix, outras: Sequence[sparse.csr_matrix], componentes: int = 200, semente: int = 11) -> list[np.ndarray]:
    """Análise semântica latente: SVD truncada do TF-IDF, ajustada no treino, com cada linha em norma 1.

    É a representação densa que não precisa de modelo de linguagem: cada avaliação vira um vetor
    de `componentes` números que resume a co-ocorrência de palavras.
    """
    svd = TruncatedSVD(n_components=componentes, random_state=semente).fit(treino)

    def unitario(m: np.ndarray) -> np.ndarray:
        norma = np.linalg.norm(m, axis=1, keepdims=True)
        return (m / np.where(norma == 0, 1.0, norma)).astype(np.float32)

    return [unitario(svd.transform(matriz)) for matriz in (treino, *outras)]


def empilhar_nota_e_texto(nota: np.ndarray, logito_do_texto: np.ndarray | None) -> np.ndarray:
    """Variáveis do modelo de segundo estágio: uma coluna por nota (1 a 5) e, se houver, o escore do texto."""
    colunas = [(nota == n).astype(float) for n in range(1, 6)]
    if logito_do_texto is not None:
        colunas.append(logito_do_texto)
    return np.column_stack(colunas)
