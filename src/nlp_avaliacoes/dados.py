"""Leitura, limpeza e separação no tempo do corpus B2W-Reviews01.

O arquivo bruto fica em data/brutos/ (fora do git). Nada aqui grava texto de avaliação em disco.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from nlp_avaliacoes import texto

RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data"
BRUTOS = DADOS / "brutos"
ARQUIVO_BRUTO = BRUTOS / "B2W-Reviews01.csv"
ARQUIVO_EMBEDDINGS = BRUTOS / "embeddings_minilm.npz"
ARQUIVO_LEITURA = DADOS / "leitura_manual_de_erros.csv"

# Conferidos em 11/11/2026 contra o arquivo do repositório de origem.
TAMANHO_ESPERADO = 49_453_175
SHA256_ESPERADO = "821fb0bf9f7230b0fba4e4f9fadd75a66d1a9ff0b1657810791d33007eb2ab38"

# Separação no tempo: o modelo aprende no começo do período, é ajustado no meio e testado no fim.
FIM_DO_TREINO = "2018-04-01"  # treino: janeiro a março
FIM_DA_VALIDACAO = "2018-05-01"  # validação e calibração: abril; teste: maio

COLUNAS = {
    "submission_date": "str",
    "reviewer_id": "str",
    "product_id": "str",
    "site_category_lv1": "category",
    "review_title": "str",
    "overall_rating": "int8",
    "recommend_to_a_friend": "category",
    "review_text": "str",
    "reviewer_birth_year": "float32",
    "reviewer_gender": "category",
    "reviewer_state": "category",
}

REGIOES = {
    "Sudeste": ["SP", "RJ", "MG", "ES"],
    "Sul": ["PR", "RS", "SC"],
    "Nordeste": ["BA", "PE", "CE", "MA", "PB", "RN", "AL", "SE", "PI"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Norte": ["PA", "AM", "RO", "TO", "AC", "AP", "RR"],
}


def ler_bruto(caminho: Path = ARQUIVO_BRUTO) -> pd.DataFrame:
    """Lê só as colunas usadas, com tipos econômicos (categorias e inteiros pequenos)."""
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe. Rode antes: uv run python scripts/baixar_dados.py")
    return pd.read_csv(caminho, usecols=list(COLUNAS), dtype=COLUNAS)


def chave_da_avaliacao(data: pd.Series, avaliador: pd.Series, produto: pd.Series) -> pd.Series:
    """Identificador curto e estável de cada avaliação (hash; não permite voltar ao avaliador)."""
    junto = data.astype(str) + "|" + avaliador.astype(str) + "|" + produto.astype(str)
    return junto.map(lambda valor: hashlib.sha1(valor.encode("utf-8")).hexdigest()[:16])


def limpar(bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove o que não serve para aprender nem para avaliar e devolve a contagem de cada descarte.

    Ordem dos descartes:
    1. sem resposta em "recomendaria" (não há o segundo rótulo);
    2. texto vazio (só título, ou nada);
    3. a mesma avaliação repetida (mesmo avaliador, mesmo produto, mesmo texto);
    4. o mesmo texto repetido por avaliadores ou produtos diferentes (modelos de texto copiados):
       fica a primeira ocorrência no tempo, para o mesmo texto não cair no treino e no teste.
    """
    tabela = bruto.copy()
    tabela["data"] = pd.to_datetime(tabela["submission_date"])
    tabela = tabela.sort_values("data", kind="stable").reset_index(drop=True)
    contagem = {"linhas_no_arquivo": int(len(tabela))}

    sem_rotulo = tabela["recommend_to_a_friend"].isna()
    contagem["sem_resposta_em_recomendaria"] = int(sem_rotulo.sum())
    tabela = tabela[~sem_rotulo]

    corpo = texto.normalizar(tabela["review_text"].fillna(""))
    vazio = corpo.str.len() == 0
    contagem["texto_vazio"] = int(vazio.sum())
    tabela, corpo = tabela[~vazio], corpo[~vazio]

    titulo = texto.normalizar(tabela["review_title"].fillna(""))
    tabela = tabela.assign(texto=(titulo + " " + corpo).str.strip(), corpo=corpo)

    repetida = tabela.duplicated(["reviewer_id", "product_id", "corpo"], keep="first")
    contagem["mesma_avaliacao_repetida"] = int(repetida.sum())
    tabela = tabela[~repetida]

    copiado = tabela.duplicated("corpo", keep="first")
    contagem["mesmo_texto_em_outra_avaliacao"] = int(copiado.sum())
    tabela = tabela[~copiado]

    limpa = pd.DataFrame(
        {
            "chave": chave_da_avaliacao(tabela["submission_date"], tabela["reviewer_id"], tabela["product_id"]),
            "data": tabela["data"],
            "avaliador": pd.factorize(tabela["reviewer_id"])[0].astype("int32"),
            "categoria": tabela["site_category_lv1"].astype(str),
            "nota": tabela["overall_rating"].astype("int8"),
            "recomenda": (tabela["recommend_to_a_friend"] == "Yes").to_numpy(),
            "texto": tabela["texto"],
            "original": (tabela["review_title"].fillna("") + ". " + tabela["review_text"]).str.strip(". "),
            "palavras": tabela["corpo"].str.count(" ").add(1).astype("int32"),
            "nascimento": tabela["reviewer_birth_year"],
            "genero": tabela["reviewer_gender"].astype(object).where(tabela["reviewer_gender"].notna(), "não informado").astype(str),
            "regiao": regiao(tabela["reviewer_state"]),
        }
    ).reset_index(drop=True)
    limpa["periodo"] = periodo(limpa["data"])
    contagem["avaliacoes_usadas"] = int(len(limpa))
    return limpa, contagem


def regiao(estado: pd.Series) -> pd.Series:
    """Região do país a partir da sigla do estado (só para tabelas agregadas)."""
    mapa = {sigla: nome for nome, siglas in REGIOES.items() for sigla in siglas}
    return estado.astype(object).map(mapa).fillna("não informada").astype(str)


def periodo(data: pd.Series) -> pd.Series:
    """Treino (até março), validação (abril) ou teste (maio)."""
    rotulo = np.where(data < FIM_DO_TREINO, "treino", np.where(data < FIM_DA_VALIDACAO, "validação", "teste"))
    return pd.Series(rotulo, index=data.index)


def carregar() -> tuple[pd.DataFrame, dict]:
    """Corpus limpo, em ordem de data, com a coluna `periodo`."""
    return limpar(ler_bruto())


def amostra_para_densa(tabela: pd.DataFrame, semente: int = 11) -> pd.DataFrame:
    """Amostra fixa em que a representação densa é calculada (o modelo de linguagem é caro demais para tudo).

    Três partes, sorteadas uma vez com semente fixa:
    - 9.000 avaliações do treino e 5.000 do teste, ao acaso: servem para comparar classificadores;
    - mais 6.000 reclamações do treino (nota até 2 e "não recomendaria"), para o agrupamento ter volume.
    Devolve as linhas com a coluna `parte` ("treino", "teste" ou "reclamações a mais").
    """
    gerador = np.random.default_rng(semente)
    treino = tabela.index[tabela["periodo"] == "treino"].to_numpy()
    teste = tabela.index[tabela["periodo"] == "teste"].to_numpy()
    sorteio_treino = gerador.choice(treino, size=min(9000, len(treino)), replace=False)
    sorteio_teste = gerador.choice(teste, size=min(5000, len(teste)), replace=False)
    reclamacao = (tabela["nota"] <= 2) & ~tabela["recomenda"] & (tabela["periodo"] == "treino")
    restantes = np.setdiff1d(tabela.index[reclamacao].to_numpy(), sorteio_treino)
    a_mais = gerador.choice(restantes, size=min(6000, len(restantes)), replace=False)
    indices = np.concatenate([sorteio_treino, sorteio_teste, a_mais])
    amostra = tabela.loc[indices].copy()
    amostra["parte"] = ["treino"] * len(sorteio_treino) + ["teste"] * len(sorteio_teste) + ["reclamações a mais"] * len(a_mais)
    return amostra


def metadados_dos_embeddings(caminho: Path = ARQUIVO_EMBEDDINGS) -> dict:
    """Como o arquivo de embeddings foi gerado (modelo, tempo, memória, tamanho do download); vazio se não existir."""
    if not caminho.exists():
        return {}
    with np.load(caminho, allow_pickle=False) as arquivo:
        return {
            "modelo": str(arquivo["modelo"]),
            **{campo: int(arquivo[campo]) for campo in ("segundos", "pico_de_memoria_mb", "download_mb", "threads") if campo in arquivo},
        }


def carregar_embeddings(chaves: pd.Series, caminho: Path = ARQUIVO_EMBEDDINGS) -> np.ndarray | None:
    """Vetores guardados por scripts/gerar_embeddings.py, na ordem de `chaves`; None se o arquivo não existir
    ou não cobrir a amostra (por exemplo, se a limpeza mudou depois que ele foi gerado)."""
    if not caminho.exists():
        return None
    with np.load(caminho, allow_pickle=False) as arquivo:
        guardadas, vetores = arquivo["chaves"], arquivo["vetores"]
    posicao = pd.Series(np.arange(len(guardadas)), index=guardadas)
    posicao = posicao[~posicao.index.duplicated()]
    alinhadas = posicao.reindex(chaves.to_numpy())
    if alinhadas.isna().any():
        return None
    return vetores[alinhadas.to_numpy().astype(int)].astype(np.float32)
