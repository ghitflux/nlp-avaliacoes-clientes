"""Motivos de reclamação: descoberta de temas, estabilidade entre execuções e dimensionamento.

A fatoração (NMF) vem do scikit-learn; o que decide se dá para confiar nos temas é a medida de
estabilidade, implementada aqui: repetir o ajuste com outra semente e outra subamostra e ver
quanto dos temas reaparece.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.exceptions import ConvergenceWarning


def palavras_do_topo(pesos: np.ndarray, nomes: np.ndarray, quantas: int = 10) -> list[list[str]]:
    """As `quantas` palavras de maior peso de cada tema (linhas de `pesos`: tema x termo)."""
    ordem = np.argsort(-pesos, axis=1, kind="stable")[:, :quantas]
    return [[str(nomes[coluna]) for coluna in linha] for linha in ordem]


def jaccard(a: Sequence, b: Sequence) -> float:
    """Tamanho da interseção dividido pelo tamanho da união de dois conjuntos (1 = iguais, 0 = nada em comum)."""
    conjunto_a, conjunto_b = set(a), set(b)
    uniao = conjunto_a | conjunto_b
    return len(conjunto_a & conjunto_b) / len(uniao) if uniao else 1.0


def semelhanca_das_palavras(pesos_a: np.ndarray, pesos_b: np.ndarray, quantas: int = 10) -> np.ndarray:
    """Matriz (temas de A x temas de B) com o Jaccard entre as `quantas` palavras de maior peso de cada tema.

    Usa só a posição das colunas, então A e B precisam ter o mesmo vocabulário.
    """
    topo_a = np.argsort(-pesos_a, axis=1, kind="stable")[:, :quantas]
    topo_b = np.argsort(-pesos_b, axis=1, kind="stable")[:, :quantas]
    return np.array([[jaccard(linha_a, linha_b) for linha_b in topo_b] for linha_a in topo_a])


def semelhanca_de_cosseno(pesos_a: np.ndarray, pesos_b: np.ndarray) -> np.ndarray:
    """Matriz (temas de A x temas de B) com o cosseno entre os vetores de pesos dos temas."""
    def unitario(m: np.ndarray) -> np.ndarray:
        norma = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norma == 0, 1.0, norma)

    return unitario(np.asarray(pesos_a, dtype=float)) @ unitario(np.asarray(pesos_b, dtype=float)).T


def emparelhar(semelhanca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Casa cada tema de A com um tema de B, um para um, maximizando a semelhança total (método húngaro).

    Devolve (par, valor): `par[i]` é o tema de B casado com o tema i de A, e `valor[i]` a semelhança do par.
    """
    linhas, colunas = linear_sum_assignment(-semelhanca)
    par = np.full(semelhanca.shape[0], -1)
    par[linhas] = colunas
    valor = np.zeros(semelhanca.shape[0])
    valor[linhas] = semelhanca[linhas, colunas]
    return par, valor


def estabilidade(pesos_a: np.ndarray, pesos_b: np.ndarray, quantas: int = 10) -> float:
    """Estabilidade entre duas execuções: média do Jaccard das palavras do topo nos pares de temas casados.

    1 significa que as duas execuções acharam os mesmos temas (em qualquer ordem); 0, que nenhum
    tema de uma tem palavra em comum com o seu par na outra.
    """
    return float(emparelhar(semelhanca_das_palavras(pesos_a, pesos_b, quantas))[1].mean())


def estabilidade_media(execucoes: Sequence[np.ndarray], quantas: int = 10) -> tuple[float, float, float]:
    """Média, mínimo e máximo da estabilidade entre todos os pares de execuções."""
    valores = [estabilidade(a, b, quantas) for a, b in combinations(execucoes, 2)]
    return float(np.mean(valores)), float(np.min(valores)), float(np.max(valores))


def estabilidade_por_tema(referencia: np.ndarray, execucoes: Sequence[np.ndarray], quantas: int = 10) -> np.ndarray:
    """Para cada tema da referência: Jaccard médio com o tema casado em cada uma das outras execuções.

    Separa os temas que sempre voltam dos que só aparecem numa execução.
    """
    return np.mean([emparelhar(semelhanca_das_palavras(referencia, e, quantas))[1] for e in execucoes], axis=0)


# ---------------------------------------------------------------------------------------------
# Ajuste
# ---------------------------------------------------------------------------------------------


def ajustar_nmf(matriz: sparse.csr_matrix, temas: int, semente: int, inicio: str = "nndsvda", iteracoes: int = 300) -> NMF:
    """NMF (fatoração em matrizes não negativas) sobre o TF-IDF: cada tema é uma lista de pesos de palavras e
    cada avaliação, uma mistura de temas. `inicio="random"` faz o resultado depender da semente."""
    modelo = NMF(n_components=temas, init=inicio, random_state=semente, max_iter=iteracoes, tol=1e-4)
    with warnings.catch_warnings():  # parar no limite de iterações é esperado aqui; a estabilidade mede o efeito
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo.fit(matriz)
    return modelo


def execucoes_de_nmf(
    matriz: sparse.csr_matrix, temas: int, repeticoes: int, semente: int, parcela: float = 0.8, iteracoes: int = 200
) -> list[np.ndarray]:
    """Repete o NMF `repeticoes` vezes, cada vez com uma subamostra de `parcela` dos documentos e
    início aleatório com semente própria. Devolve a matriz tema x termo de cada execução."""
    gerador = np.random.default_rng(semente)
    execucoes = []
    for repeticao in range(repeticoes):
        linhas = gerador.choice(matriz.shape[0], size=int(parcela * matriz.shape[0]), replace=False)
        modelo = ajustar_nmf(matriz[linhas], temas, semente=semente * 1000 + repeticao, inicio="random", iteracoes=iteracoes)
        execucoes.append(modelo.components_)
    return execucoes


def tema_dominante(mistura: np.ndarray) -> np.ndarray:
    """Índice do tema de maior peso em cada avaliação; -1 quando nenhum tema tem peso (texto sem termo conhecido)."""
    dominante = np.argmax(mistura, axis=1)
    dominante[mistura.sum(axis=1) <= 0] = -1
    return dominante


def parcelas(dominante: np.ndarray, temas: int) -> np.ndarray:
    """Parcela das avaliações em cada tema (as sem tema ficam fora do numerador, dentro do denominador)."""
    return np.bincount(dominante[dominante >= 0], minlength=temas) / len(dominante)


def palavras_caracteristicas(tfidf: sparse.csr_matrix, grupo: np.ndarray, grupos: int) -> np.ndarray:
    """Para comparar agrupamento com NMF na mesma régua: transforma cada grupo de avaliações numa linha de
    pesos de palavras (TF-IDF médio do grupo menos o TF-IDF médio geral, sem negativos)."""
    geral = np.asarray(tfidf.mean(axis=0)).ravel()
    pesos = np.zeros((grupos, tfidf.shape[1]))
    for g in range(grupos):
        membros = grupo == g
        if membros.any():
            pesos[g] = np.clip(np.asarray(tfidf[membros].mean(axis=0)).ravel() - geral, 0.0, None)
    return pesos


def execucoes_de_kmeans(
    vetores: np.ndarray, tfidf: sparse.csr_matrix, grupos: int, repeticoes: int, semente: int, parcela: float = 0.8
) -> list[np.ndarray]:
    """O mesmo protocolo de `execucoes_de_nmf` para o agrupamento k-means sobre a representação densa:
    subamostra de `parcela`, semente própria, e cada grupo descrito pelas suas palavras características."""
    gerador = np.random.default_rng(semente)
    execucoes = []
    for repeticao in range(repeticoes):
        linhas = gerador.choice(vetores.shape[0], size=int(parcela * vetores.shape[0]), replace=False)
        modelo = KMeans(n_clusters=grupos, n_init=1, random_state=semente * 1000 + repeticao).fit(vetores[linhas])
        execucoes.append(palavras_caracteristicas(tfidf[linhas], modelo.labels_, grupos))
    return execucoes


# ---------------------------------------------------------------------------------------------
# Nomes
# ---------------------------------------------------------------------------------------------


def nomear(topo: Sequence[Sequence[str]], ancoras: dict[str, Sequence[str]], minimo: int = 2, sem_nome: str = "Outros") -> list[str]:
    """Dá a cada tema o nome do motivo cujas palavras-âncora mais pesam entre as palavras do topo do tema.

    Os motivos e as âncoras são escritos à mão depois de ler as palavras de cada tema; a regra só
    torna a atribuição reprodutível e aplicável a qualquer execução. Uma âncora na 1ª posição do
    tema vale mais que na última (peso = número de palavras do topo menos a posição). Dois temas
    podem receber o mesmo motivo (o NMF costuma separar "não recebi" de "não chegou"). Tema com
    menos de `minimo` âncoras de qualquer motivo fica como `sem_nome`.
    """
    nomes = []
    for palavras in topo:
        melhor, melhor_peso = sem_nome, 0
        for nome, lista in ancoras.items():
            posicoes = [posicao for posicao, palavra in enumerate(palavras) if palavra in lista]
            peso = sum(len(palavras) - posicao for posicao in posicoes)
            if len(posicoes) >= minimo and peso > melhor_peso:
                melhor, melhor_peso = nome, peso
        nomes.append(melhor)
    return nomes


def mistura(matriz: sparse.csr_matrix, pesos: np.ndarray, iteracoes: int = 200) -> np.ndarray:
    """Mistura de temas de cada avaliação para temas já ajustados (`pesos`: tema x termo), sem reajustá-los.

    É o que permite medir, nas avaliações de teste, o tamanho dos temas aprendidos no treino.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        misturas, _, _ = non_negative_factorization(
            matriz, H=np.asarray(pesos, dtype=matriz.dtype), n_components=pesos.shape[0], init="custom", update_H=False, max_iter=iteracoes, tol=1e-4
        )
    return misturas


def parcela_por_motivo(dominante: np.ndarray, motivo_do_tema: Sequence[str], motivos: Sequence[str]) -> np.ndarray:
    """Parcela das avaliações cujo tema dominante pertence a cada motivo (na ordem de `motivos`)."""
    por_tema = parcelas(dominante, len(motivo_do_tema))
    return np.array([sum(p for p, nome in zip(por_tema, motivo_do_tema) if nome == motivo) for motivo in motivos])
