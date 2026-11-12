"""Do texto ao vetor: normalização, contagem de termos e TF-IDF, implementados em NumPy/SciPy.

O TF-IDF daqui segue a mesma fórmula do scikit-learn (IDF suavizado, norma L2), o que permite
conferir uma implementação contra a outra num corpus mínimo (ver tests/test_texto.py).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

# Palavras de função, sem acento (o texto é normalizado antes). "nao", "nunca", "nem", "sem" e "mas"
# ficam de fora da lista de propósito: negação e ressalva carregam o sentido de uma reclamação.
PALAVRAS_VAZIAS = frozenset(
    """a o as os um uma uns umas de do da dos das em no na nos nas por para pra pro com e ou que se
    ao aos pela pelo pelas pelos este esta estes estas esse essa esses essas isso isto aquele aquela aquilo
    eu me meu minha meus minhas voce voces ele ela eles elas nos lhe te seu sua seus suas dele dela
    e foi ser sao era estao esta estou estava tem tinha ter tenho ja so tambem muito muita mais como quando
    onde ate entre sobre depois antes ainda la aqui ai entao pois porque que qual quem cada todo toda todos todas
    vai vou fui ha foram sera sido""".split()
)

# Léxico pequeno, escrito à mão antes de olhar qualquer resultado: palavras e pares de palavras que
# um leitor associaria a reclamação. É a linha de base sem treino.
LEXICO_NEGATIVO = frozenset(
    """pessimo pessima ruim ruins horrivel horriveis terrivel lixo porcaria pior decepcao decepcionado decepcionada
    decepcionante decepcionei insatisfeito insatisfeita arrependido arrependida arrependi arrependimento
    defeito defeituoso quebrado quebrada quebrou quebra parou estragou estragado danificado danificada
    amassado amassada rasgado rasgada trincado trincada riscado riscada falso falsificado enganosa enganacao
    fraco fraca fragil atraso atrasado atrasada atrasou demora demorou demorada errado errada faltando faltou
    incompleto incompleta devolver devolucao devolvi reembolso estorno cancelar cancelei cancelamento
    reclamacao reclamar procon problema problemas descaso vergonha absurdo desrespeito prejuizo propaganda
    nao_recebi nao_chegou nao_funciona nao_funcionou nao_recomendo nao_gostei nao_comprem nao_compre
    nao_compro nao_vale nao_entregaram nao_entregue nao_presta nao_liga nao_veio nao_serve nao_carrega
    nunca_mais ate_agora ate_hoje ma_qualidade baixa_qualidade sem_resposta""".split()
)


def normalizar(textos: pd.Series) -> pd.Series:
    """Minúsculas, sem acento, só letras e espaço; letra repetida três vezes ou mais vira uma só.

    "Péssimooo!!! Não   recebi." vira "pessimo nao recebi". Tirar o acento junta "não" com "nao"
    e "ótimo" com "otimo", que nas avaliações aparecem nas duas formas.
    """
    sem_acento = textos.astype(str).str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    so_letras = sem_acento.str.replace(r"[^a-z]+", " ", regex=True).str.replace(r"([a-z])\1{2,}", r"\1", regex=True)
    return so_letras.str.strip()


def termos(texto_normalizado: str, pares: bool = True, ignorar: frozenset[str] = frozenset()) -> list[str]:
    """Palavras do texto e, se `pares`, os pares de palavras vizinhas unidos por "_" (bigramas).

    As palavras em `ignorar` saem antes de formar os pares.
    """
    palavras = [p for p in texto_normalizado.split() if p not in ignorar]
    if not pares:
        return palavras
    return palavras + [f"{a}_{b}" for a, b in zip(palavras, palavras[1:])]


def termos_com_negacao(texto_normalizado: str, ignorar: frozenset[str] = frozenset()) -> list[str]:
    """Palavras do texto com o "nao" colado na palavra seguinte: "nao recebi o produto" vira
    ["nao_recebi", "produto"] (com "o" em `ignorar`).

    Serve para os temas: "recebi" e "nao_recebi" passam a ser termos diferentes, sem a redundância
    de carregar a palavra, o par e a negação solta ao mesmo tempo.
    """
    palavras = [p for p in texto_normalizado.split() if p not in ignorar]
    saida, i = [], 0
    while i < len(palavras):
        if palavras[i] == "nao" and i + 1 < len(palavras):
            saida.append(f"nao_{palavras[i + 1]}")
            i += 2
        else:
            saida.append(palavras[i])
            i += 1
    return saida


def escore_do_lexico(textos_normalizados: Iterable[str], lexico: frozenset[str] = LEXICO_NEGATIVO) -> np.ndarray:
    """Linha de base sem treino: quantos termos do léxico negativo aparecem em cada texto."""
    return np.array([sum(t in lexico for t in termos(texto)) for texto in textos_normalizados], dtype=float)


# ---------------------------------------------------------------------------------------------
# Contagens
# ---------------------------------------------------------------------------------------------


def montar_vocabulario(
    termos_por_documento: Sequence[Sequence[str]], df_minimo: int = 1, maximo: int | None = None, df_maximo: float = 1.0
) -> dict[str, int]:
    """Termos que aparecem em pelo menos `df_minimo` documentos, em ordem alfabética (termo -> coluna).

    `df_maximo` descarta os termos presentes em mais do que essa parcela dos documentos (palavras
    que estão em quase tudo não distinguem nada). Com `maximo`, ficam os termos presentes em mais documentos.
    """
    frequencia: Counter[str] = Counter()
    for lista in termos_por_documento:
        frequencia.update(set(lista))
    teto = df_maximo * len(termos_por_documento)
    mantidos = [(termo, df) for termo, df in frequencia.items() if df_minimo <= df <= teto]
    if maximo is not None and len(mantidos) > maximo:
        mantidos = sorted(mantidos, key=lambda par: (-par[1], par[0]))[:maximo]
    return {termo: coluna for coluna, termo in enumerate(sorted(termo for termo, _ in mantidos))}


def contar(termos_por_documento: Sequence[Sequence[str]], vocabulario: dict[str, int]) -> sparse.csr_matrix:
    """Matriz esparsa documento x termo com o número de ocorrências; termos fora do vocabulário são ignorados."""
    linhas, colunas = [], []
    for linha, lista in enumerate(termos_por_documento):
        for termo in lista:
            coluna = vocabulario.get(termo)
            if coluna is not None:
                linhas.append(linha)
                colunas.append(coluna)
    valores = np.ones(len(linhas), dtype=np.float64)
    forma = (len(termos_por_documento), len(vocabulario))
    matriz = sparse.coo_matrix((valores, (linhas, colunas)), shape=forma).tocsr()  # soma as repetições
    matriz.sort_indices()
    return matriz


# N-gramas de caracteres: cada letra vira um dígito em base 27 (espaço = 0, a = 1, ..., z = 26) e
# cada n-grama vira um número inteiro. Assim a contagem inteira é feita com operações de vetor.
BASE = 27


def _deslocamento(n: int, n_minimo: int) -> int:
    """Primeiro código reservado aos n-gramas de tamanho n (os tamanhos não se misturam)."""
    return sum(BASE**m for m in range(n_minimo, n))


def codigos_de_caracteres(textos_normalizados: Sequence[str], n_minimo: int = 3, n_maximo: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Todos os n-gramas de caracteres do lote, como pares (documento, código do n-grama).

    Cada texto recebe um espaço no começo e no fim, para o início e o fim das palavras da borda
    também formarem n-gramas. N-gramas que cruzariam dois documentos são descartados.
    """
    com_borda = [f" {texto} " for texto in textos_normalizados]
    tamanhos = np.array([len(t) for t in com_borda], dtype=np.int64)
    bruto = np.frombuffer("".join(com_borda).encode("ascii"), dtype=np.uint8)
    digitos = np.where(bruto == 32, 0, bruto.astype(np.int64) - 96)
    if digitos.size and (digitos.min() < 0 or digitos.max() > 26):
        raise ValueError("O texto precisa estar normalizado (só letras de a a z e espaço).")
    documento = np.repeat(np.arange(len(com_borda), dtype=np.int64), tamanhos)

    documentos, codigos = [], []
    for n in range(n_minimo, n_maximo + 1):
        posicoes = digitos.size - n + 1
        if posicoes <= 0:
            continue
        valor = np.zeros(posicoes, dtype=np.int64)
        for j in range(n):
            valor = valor * BASE + digitos[j : j + posicoes]
        dentro = documento[:posicoes] == documento[n - 1 :]
        documentos.append(documento[:posicoes][dentro])
        codigos.append(valor[dentro] + _deslocamento(n, n_minimo))
    if not documentos:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(documentos), np.concatenate(codigos)


def decodificar(codigo: int, n_minimo: int = 3, n_maximo: int = 5) -> str:
    """O n-grama de caracteres que corresponde a um código (inverso de `codigos_de_caracteres`)."""
    for n in range(n_maximo, n_minimo - 1, -1):
        inicio = _deslocamento(n, n_minimo)
        if codigo >= inicio:
            resto, letras = codigo - inicio, []
            for _ in range(n):
                resto, digito = divmod(resto, BASE)
                letras.append(" " if digito == 0 else chr(96 + digito))
            return "".join(reversed(letras))
    raise ValueError("código inválido")


def contar_caracteres(
    textos_normalizados: Sequence[str],
    vocabulario: np.ndarray | None = None,
    df_minimo: int = 1,
    n_minimo: int = 3,
    n_maximo: int = 5,
    lote: int = 20000,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Matriz documento x n-grama de caracteres e o vocabulário (códigos em ordem crescente).

    Sem `vocabulario`, ele é aprendido aqui: os n-gramas presentes em pelo menos `df_minimo`
    documentos. Com `vocabulario`, só esses n-gramas são contados (é o caso do teste). O trabalho
    é feito em lotes para a memória não crescer com o corpus.
    """
    total = len(textos_normalizados)
    linhas, codigos, contagens = [], [], []
    for inicio in range(0, total, lote):
        documento, codigo = codigos_de_caracteres(textos_normalizados[inicio : inicio + lote], n_minimo, n_maximo)
        if vocabulario is not None:  # descarta cedo o que não está no vocabulário
            posicao = np.searchsorted(vocabulario, codigo)
            posicao[posicao == len(vocabulario)] = 0
            presente = vocabulario[posicao] == codigo if len(vocabulario) else np.zeros(len(codigo), dtype=bool)
            documento, codigo = documento[presente], codigo[presente]
        espaco = _deslocamento(n_maximo + 1, n_minimo)
        par, quantos = np.unique(documento * espaco + codigo, return_counts=True)
        linhas.append(par // espaco + inicio)
        codigos.append(par % espaco)
        contagens.append(quantos)
    linhas_, codigos_, contagens_ = (np.concatenate(x) if x else np.zeros(0, dtype=np.int64) for x in (linhas, codigos, contagens))

    if vocabulario is None:
        unicos, df = np.unique(codigos_, return_counts=True)
        vocabulario = unicos[df >= df_minimo]
        presente = np.isin(codigos_, vocabulario)
        linhas_, codigos_, contagens_ = linhas_[presente], codigos_[presente], contagens_[presente]
    colunas = np.searchsorted(vocabulario, codigos_)
    matriz = sparse.csr_matrix((contagens_.astype(np.float64), (linhas_, colunas)), shape=(total, len(vocabulario)))
    matriz.sort_indices()
    return matriz, vocabulario


# ---------------------------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------------------------


def idf_suavizado(contagens: sparse.csr_matrix) -> np.ndarray:
    """IDF de cada termo: ln((1 + n) / (1 + df)) + 1, com n documentos e df documentos que contêm o termo.

    O "+1" dentro do logaritmo age como um documento a mais que contém todos os termos (evita
    divisão por zero); o "+1" de fora impede que um termo presente em tudo tenha peso zero.
    """
    n = contagens.shape[0]
    df = np.bincount(contagens.indices, minlength=contagens.shape[1])
    return np.log((1.0 + n) / (1.0 + df)) + 1.0


def ponderar(contagens: sparse.csr_matrix, idf: np.ndarray, sublinear: bool = False) -> sparse.csr_matrix:
    """TF-IDF: contagem (ou 1 + ln da contagem, se `sublinear`) vezes o IDF, com cada linha em norma L2 igual a 1."""
    matriz = contagens.astype(np.float64, copy=True)
    if sublinear:
        matriz.data = 1.0 + np.log(matriz.data)
    matriz.data *= idf[matriz.indices]
    norma = np.sqrt(np.asarray(matriz.multiply(matriz).sum(axis=1)).ravel())
    norma[norma == 0] = 1.0
    matriz.data /= np.repeat(norma, np.diff(matriz.indptr))
    return matriz


@dataclass
class TfidfDePalavras:
    """TF-IDF de palavras. O vocabulário e o IDF são aprendidos só em `ajustar`.

    `modo`: "pares" (palavras e pares de palavras vizinhas, para o classificador), "palavras" (só
    palavras) ou "negacao" (palavras com o "nao" colado na seguinte, para os temas).
    """

    df_minimo: int = 5
    maximo: int | None = None
    modo: str = "pares"
    sublinear: bool = True
    ignorar: frozenset[str] = frozenset()
    df_maximo: float = 1.0
    vocabulario: dict[str, int] | None = None
    idf: np.ndarray | None = None

    def _termos(self, textos: Sequence[str]) -> list[list[str]]:
        if self.modo == "negacao":
            return [termos_com_negacao(t, self.ignorar) for t in textos]
        return [termos(t, self.modo == "pares", self.ignorar) for t in textos]

    def ajustar(self, textos: Sequence[str]) -> sparse.csr_matrix:
        listas = self._termos(textos)
        self.vocabulario = montar_vocabulario(listas, self.df_minimo, self.maximo, self.df_maximo)
        contagens = contar(listas, self.vocabulario)
        self.idf = idf_suavizado(contagens)
        return ponderar(contagens, self.idf, self.sublinear)

    def transformar(self, textos: Sequence[str]) -> sparse.csr_matrix:
        if self.vocabulario is None or self.idf is None:
            raise RuntimeError("Chame `ajustar` com os textos de treino antes de `transformar`.")
        return ponderar(contar(self._termos(textos), self.vocabulario), self.idf, self.sublinear)

    @property
    def nomes(self) -> np.ndarray:
        return np.array(sorted(self.vocabulario, key=self.vocabulario.get))


@dataclass
class TfidfDeCaracteres:
    """TF-IDF de n-gramas de caracteres (3 a 5 letras por padrão): tolera erro de digitação e flexão."""

    df_minimo: int = 5
    n_minimo: int = 3
    n_maximo: int = 5
    sublinear: bool = True
    vocabulario: np.ndarray | None = None
    idf: np.ndarray | None = None

    def ajustar(self, textos: Sequence[str]) -> sparse.csr_matrix:
        contagens, self.vocabulario = contar_caracteres(textos, None, self.df_minimo, self.n_minimo, self.n_maximo)
        self.idf = idf_suavizado(contagens)
        return ponderar(contagens, self.idf, self.sublinear)

    def transformar(self, textos: Sequence[str]) -> sparse.csr_matrix:
        if self.vocabulario is None or self.idf is None:
            raise RuntimeError("Chame `ajustar` com os textos de treino antes de `transformar`.")
        contagens, _ = contar_caracteres(textos, self.vocabulario, 1, self.n_minimo, self.n_maximo)
        return ponderar(contagens, self.idf, self.sublinear)
