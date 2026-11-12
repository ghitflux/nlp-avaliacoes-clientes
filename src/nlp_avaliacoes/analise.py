"""Cada tabela do README e do notebook sai de uma função daqui.

Convenção: y = 1 quando a avaliação é uma reclamação segundo o rótulo em uso, e o escore de
qualquer método cresce com a suspeita de reclamação. O rótulo principal é a resposta "não" à
pergunta "recomendaria este produto?"; a nota de 1 a 5 entra como segundo rótulo e como linha de base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from nlp_avaliacoes import metricas, modelos, temas, texto

SEMENTE = 11
REPETICOES = 1000
PARCELA_LIDA = 0.10
ALVO_DE_CAPTURA = 0.80
FRACOES = np.round(np.linspace(0.0, 1.0, 41), 3)
TERMOS_DE_ENTREGA = r"\b(?:entreg\w*|prazo|atras\w+|cheg\w+|receb\w+|transportadora|correios|rastre\w+)\b"


def rotulo(tabela: pd.DataFrame, qual: str = "recomendaria") -> np.ndarray:
    """Vetor de reclamação segundo o rótulo pedido: "recomendaria" (respondeu não) ou "nota" (nota até 2)."""
    if qual == "recomendaria":
        return (~tabela["recomenda"]).to_numpy()
    if qual == "nota":
        return (tabela["nota"] <= 2).to_numpy()
    raise ValueError(qual)


# ---------------------------------------------------------------------------------------------
# A base e os dois rótulos
# ---------------------------------------------------------------------------------------------


def resumo_da_base(tabela: pd.DataFrame, contagem: dict) -> dict:
    partes = {nome: tabela[tabela["periodo"] == nome] for nome in ("treino", "validação", "teste")}
    return {
        **contagem,
        "primeira_data": str(tabela["data"].min().date()),
        "ultima_data": str(tabela["data"].max().date()),
        "categorias": int(tabela["categoria"].nunique()),
        "mediana_de_palavras": float(tabela["palavras"].median()),
        "periodos": {
            nome: {
                "avaliacoes": int(len(parte)),
                "de": str(parte["data"].min().date()),
                "ate": str(parte["data"].max().date()),
                "nao_recomendaria": float((~parte["recomenda"]).mean()),
                "nota_ate_2": float((parte["nota"] <= 2).mean()),
            }
            for nome, parte in partes.items()
        },
        "avaliadores_do_teste_vistos_no_treino": float(partes["teste"]["avaliador"].isin(partes["treino"]["avaliador"]).mean()),
    }


def discordancia_por_nota(tabela: pd.DataFrame) -> pd.DataFrame:
    """Para cada nota: quantas avaliações e que parcela respondeu que não recomendaria (intervalo de Wilson)."""
    linhas = []
    for nota, grupo in tabela.groupby("nota"):
        nao, total = int((~grupo["recomenda"]).sum()), len(grupo)
        inferior, superior = metricas.intervalo_de_wilson(nao, total)
        linhas.append({"nota": int(nota), "avaliações": total, "não recomendaria": nao / total, "IC 95% inferior": inferior, "IC 95% superior": superior})
    return pd.DataFrame(linhas).set_index("nota")


def resumo_da_discordancia(tabela: pd.DataFrame) -> dict:
    """Onde a nota e o "recomendaria" contam histórias diferentes, com intervalo de Wilson em cada parcela."""
    nota, nao = tabela["nota"].to_numpy(), (~tabela["recomenda"]).to_numpy()

    def parcela(caso: np.ndarray, universo: np.ndarray) -> dict:
        sucessos, total = int((caso & universo).sum()), int(universo.sum())
        inferior, superior = metricas.intervalo_de_wilson(sucessos, total)
        return {"avaliacoes": sucessos, "de": total, "parcela": sucessos / total, "inferior": inferior, "superior": superior}

    todos = np.ones(len(tabela), dtype=bool)
    return {
        "rotulos_discordam": parcela((nota <= 2) != nao, todos),
        "nota_baixa_mas_recomendaria": parcela(~nao, nota <= 2),
        "nota_alta_mas_nao_recomendaria": parcela(nao, nota >= 4),
        "nota_3_e_nao_recomendaria": parcela(nao, nota == 3),
        "nao_recomendaria_com_nota_3_ou_mais": parcela(nota >= 3, nao),
    }


# ---------------------------------------------------------------------------------------------
# O alerta: quanto cada método acerta
# ---------------------------------------------------------------------------------------------

COLUNAS_DO_ALERTA = ["AUC", "precisão lendo 10%", "revocação lendo 10%", "leitura para pegar 80%"]


def metricas_do_alerta(y: np.ndarray, escore: np.ndarray) -> np.ndarray:
    """[AUC, precisão e revocação lendo os 10% de maior escore, parcela a ler para pegar 80% das reclamações]."""
    fracoes = np.concatenate([[PARCELA_LIDA], metricas.GRADE_DE_LEITURA])
    area, revocacao, precisao = metricas.auc_e_capturas(y, escore, fracoes)  # uma ordenação só para as quatro métricas
    leitura = metricas.GRADE_DE_LEITURA[np.argmax(revocacao[1:] >= ALVO_DE_CAPTURA - 1e-12)]
    return np.array([area, precisao[0], revocacao[0], leitura])


def desempenho(
    y: np.ndarray, escores: dict[str, np.ndarray], pares: list[tuple[str, str]] | None = None, repeticoes: int = REPETICOES, semente: int = SEMENTE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas do alerta por método e diferenças entre pares de métodos, com intervalo de 95%.

    Um único bootstrap reamostra as avaliações e recalcula todos os métodos nas mesmas
    avaliações sorteadas, então a diferença entre dois métodos é pareada: se o intervalo dela
    inclui o zero, os dois empatam naquela métrica. Devolve (tabela por método, tabela de diferenças).
    """
    nomes = list(escores)
    ponto = np.array([metricas_do_alerta(y, escores[nome]) for nome in nomes])
    amostras = metricas.reamostragens(
        lambda y_, *e: np.array([metricas_do_alerta(y_, escore) for escore in e]), y, *escores.values(), repeticoes=repeticoes, semente=semente
    )  # repetições x métodos x métricas
    inferior, superior = metricas.intervalo(amostras)
    linhas = {}
    for m, nome in enumerate(nomes):
        linhas[nome] = {}
        for i, coluna in enumerate(COLUNAS_DO_ALERTA):
            linhas[nome].update({coluna: ponto[m, i], f"{coluna} inf": inferior[m, i], f"{coluna} sup": superior[m, i]})
    comparacoes = {}
    for a, b in pares or []:
        ia, ib = nomes.index(a), nomes.index(b)
        inferior_d, superior_d = metricas.intervalo(amostras[:, ia, :] - amostras[:, ib, :])
        comparacoes[f"{a} − {b}"] = {}
        for i, coluna in enumerate(COLUNAS_DO_ALERTA):
            comparacoes[f"{a} − {b}"].update(
                {coluna: ponto[ia, i] - ponto[ib, i], f"{coluna} inf": inferior_d[i], f"{coluna} sup": superior_d[i], f"{coluna} empata": bool(inferior_d[i] <= 0.0 <= superior_d[i])}
            )
    return pd.DataFrame.from_dict(linhas, orient="index"), pd.DataFrame.from_dict(comparacoes, orient="index")


def curvas_de_captura(y: np.ndarray, escores: dict[str, np.ndarray], com_faixa: str, repeticoes: int = REPETICOES, semente: int = SEMENTE) -> pd.DataFrame:
    """Parcela das reclamações capturadas para cada parcela lida, por método, mais o teto (ordenação perfeita)
    e a faixa de 95% do método `com_faixa`."""
    curvas = pd.DataFrame({"parcela lida": FRACOES})
    for nome, escore in escores.items():
        curvas[nome] = metricas.curva_de_captura(y, escore, FRACOES)
    curvas["acaso"] = FRACOES
    curvas["teto"] = np.minimum(1.0, FRACOES / y.mean())
    inferior, superior = metricas.bootstrap(
        lambda y_, e_: metricas.curva_de_captura(y_, e_, FRACOES), y, escores[com_faixa], repeticoes=repeticoes, semente=semente
    )
    curvas[f"{com_faixa} inf"], curvas[f"{com_faixa} sup"] = inferior, superior
    return curvas


def regra_da_nota(y: np.ndarray, nota: np.ndarray, escore_do_texto: np.ndarray, repeticoes: int = REPETICOES, semente: int = SEMENTE) -> pd.DataFrame:
    """A regra "nota até 2" contra o rótulo, e o modelo de texto marcando o mesmo número de avaliações."""
    regra = nota <= 2
    volume = float(regra.mean())

    def da_regra(y_, regra_):
        return np.array(metricas.precisao_e_revocacao(y_, regra_))

    def do_texto(y_, escore_):
        revocacao, precisao = metricas.captura_no_topo(y_, escore_, volume)
        return np.array([precisao, revocacao])

    linhas = {}
    for nome, funcao, vetor in (('Regra "nota até 2"', da_regra, regra), ("Texto, marcando o mesmo volume", do_texto, escore_do_texto)):
        ponto = funcao(y, vetor)
        inferior, superior = metricas.bootstrap(funcao, y, vetor, repeticoes=repeticoes, semente=semente)
        linhas[nome] = {
            "parcela marcada": volume,
            "precisão": ponto[0], "precisão inf": inferior[0], "precisão sup": superior[0],
            "revocação": ponto[1], "revocação inf": inferior[1], "revocação sup": superior[1],
        }
    return pd.DataFrame.from_dict(linhas, orient="index")


# ---------------------------------------------------------------------------------------------
# Ruído de rótulo
# ---------------------------------------------------------------------------------------------


def auc_por_rotulo(teste: pd.DataFrame, escores_por_treino: dict[str, np.ndarray], repeticoes: int = REPETICOES, semente: int = SEMENTE) -> pd.DataFrame:
    """A AUC do mesmo escore conforme o rótulo que serve de gabarito.

    `escores_por_treino` traz o escore de teste do modelo treinado com cada rótulo. Gabaritos:
    cada rótulo sozinho; só as avaliações em que os dois concordam (o caso limpo); e "qualquer
    um dos dois sinais" (nota até 2 ou não recomendaria).
    """
    nao, baixa = rotulo(teste, "recomendaria"), rotulo(teste, "nota")
    concordam = nao == baixa
    gabaritos = {
        "não recomendaria": (np.ones(len(teste), dtype=bool), nao),
        "nota até 2": (np.ones(len(teste), dtype=bool), baixa),
        "só onde os dois concordam": (concordam, nao),
        "qualquer um dos dois sinais": (np.ones(len(teste), dtype=bool), nao | baixa),
    }

    def as_quatro(nao_, baixa_, escore_) -> np.ndarray:
        postos = stats.rankdata(escore_)  # os três gabaritos sobre todas as avaliações usam os mesmos postos
        iguais = nao_ == baixa_
        return np.array(
            [
                metricas.auc_dos_postos(nao_, postos), metricas.auc_dos_postos(baixa_, postos),
                metricas.auc(nao_[iguais], escore_[iguais]), metricas.auc_dos_postos(nao_ | baixa_, postos),
            ]
        )

    linhas = []
    for treinado, escore in escores_por_treino.items():
        ponto = as_quatro(nao, baixa, escore)
        inferior, superior = metricas.bootstrap(as_quatro, nao, baixa, escore, repeticoes=repeticoes, semente=semente)
        for i, (gabarito, (recorte, y)) in enumerate(gabaritos.items()):
            linhas.append(
                {
                    "treinado com": treinado, "avaliado contra": gabarito, "avaliações": int(recorte.sum()),
                    "reclamações": float(y[recorte].mean()), "AUC": ponto[i], "IC 95% inferior": inferior[i], "IC 95% superior": superior[i],
                }
            )
    return pd.DataFrame(linhas).set_index(["treinado com", "avaliado contra"])


def texto_dentro_da_nota(teste: pd.DataFrame, escore: np.ndarray, repeticoes: int = REPETICOES, semente: int = SEMENTE) -> pd.DataFrame:
    """Dentro de cada nota, o texto ainda separa quem não recomendaria? AUC do escore de texto por nota."""
    nao, nota = rotulo(teste, "recomendaria"), teste["nota"].to_numpy()
    linhas = []
    for n in range(1, 6):
        recorte = nota == n
        inferior, superior = metricas.bootstrap(metricas.auc, nao[recorte], escore[recorte], repeticoes=repeticoes, semente=semente)
        linhas.append(
            {
                "nota": n, "avaliações": int(recorte.sum()), "não recomendaria": float(nao[recorte].mean()),
                "AUC do texto": metricas.auc(nao[recorte], escore[recorte]), "IC 95% inferior": inferior, "IC 95% superior": superior,
            }
        )
    return pd.DataFrame(linhas).set_index("nota")


def nota_mais_texto(
    validacao: pd.DataFrame, logito_validacao: np.ndarray, teste: pd.DataFrame, logito_teste: np.ndarray
) -> dict[str, np.ndarray]:
    """Escores de teste de dois modelos de segundo estágio, ajustados na validação: só a nota, e nota mais texto.

    O modelo de texto foi treinado antes, no treino; aqui uma regressão logística pequena (seis
    pesos) aprende quanto o escore do texto acrescenta a quem já sabe a nota.
    """
    y = rotulo(validacao, "recomendaria")
    escores = {}
    for nome, logito_v, logito_t in (("Só a nota", None, None), ("Nota + texto", logito_validacao, logito_teste)):
        entrada = modelos.empilhar_nota_e_texto(validacao["nota"].to_numpy(), logito_v)
        modelo = LogisticRegression(C=1e4, solver="lbfgs", fit_intercept=False, max_iter=2000).fit(entrada, y)
        escores[nome] = modelo.decision_function(modelos.empilhar_nota_e_texto(teste["nota"].to_numpy(), logito_t))
    return escores


def texto_negativo_com_nota_alta(teste: pd.DataFrame, probabilidade: np.ndarray) -> dict:
    """Entre as avaliações de nota 4 ou 5 que recomendariam: quantas o modelo lê como reclamação."""
    alta = (teste["nota"] >= 4).to_numpy() & teste["recomenda"].to_numpy()
    marcadas = int((probabilidade[alta] >= 0.5).sum())
    inferior, superior = metricas.intervalo_de_wilson(marcadas, int(alta.sum()))
    return {"avaliacoes": int(alta.sum()), "lidas_como_reclamacao": marcadas, "parcela": marcadas / alta.sum(), "inferior": inferior, "superior": superior}


# ---------------------------------------------------------------------------------------------
# Calibração e erros
# ---------------------------------------------------------------------------------------------


def calibracao(
    y_validacao: np.ndarray, logito_validacao: np.ndarray, y_teste: np.ndarray, logito_teste: np.ndarray,
    repeticoes: int = REPETICOES, semente: int = SEMENTE,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Compara a probabilidade bruta do modelo com a recalibrada por Platt (ajustada na validação).

    Devolve o resumo (erro de calibração e Brier de cada versão, com intervalo), a curva de
    calibração por faixas e a probabilidade recalibrada do teste.
    """
    a, b = metricas.ajustar_platt(logito_validacao, y_validacao)
    versoes = {"bruta": metricas.sigmoide(logito_teste), "recalibrada": metricas.sigmoide(a * logito_teste + b)}
    resumo: dict = {"platt_a": a, "platt_b": b}
    curvas = []
    amostras = metricas.reamostragens(
        lambda y_, p_, q_: np.array([metricas.erro_de_calibracao(y_, p_), metricas.erro_de_calibracao(y_, q_)]),
        y_teste, versoes["bruta"], versoes["recalibrada"], repeticoes=repeticoes, semente=semente,
    )  # as duas versões nas mesmas reamostragens: a diferença é pareada
    inferior, superior = metricas.intervalo(amostras)
    for i, (nome, probabilidade) in enumerate(versoes.items()):
        resumo[nome] = {
            "erro_de_calibracao": metricas.erro_de_calibracao(y_teste, probabilidade), "inferior": float(inferior[i]), "superior": float(superior[i]),
            "brier": metricas.brier(y_teste, probabilidade),
            "probabilidade_media": float(probabilidade.mean()), "frequencia_observada": float(y_teste.mean()),
        }
        prevista, observada, tamanho = metricas.curva_de_calibracao(y_teste, probabilidade, 10)
        curvas.append(pd.DataFrame({"versão": nome, "faixa": np.arange(1, 11), "prevista": prevista, "observada": observada, "avaliações": tamanho}))
    diferenca = metricas.intervalo(amostras[:, 1] - amostras[:, 0])
    resumo["diferenca_recalibrada_menos_bruta"] = {
        "valor": resumo["recalibrada"]["erro_de_calibracao"] - resumo["bruta"]["erro_de_calibracao"], "inferior": diferenca[0], "superior": diferenca[1],
    }
    return resumo, pd.concat(curvas, ignore_index=True), versoes["recalibrada"]


def faixa_de_tamanho(palavras: pd.Series) -> pd.Series:
    """Tamanho do texto em faixas de número de palavras."""
    return pd.cut(palavras, [0, 5, 10, 20, 40, np.inf], labels=["até 5 palavras", "6 a 10", "11 a 20", "21 a 40", "mais de 40"]).astype(str)


def faixa_de_idade(nascimento: pd.Series, ano: int = 2018) -> pd.Series:
    """Idade em 2018, em faixas; anos de nascimento implausíveis ficam como "não informada"."""
    idade = ano - nascimento
    faixa = pd.cut(idade, [17, 29, 39, 49, 59, 100], labels=["18 a 29", "30 a 39", "40 a 49", "50 a 59", "60 ou mais"]).astype(object)
    return faixa.where(faixa.notna(), "não informada").astype(str)


def erros_por_recorte(
    teste: pd.DataFrame, y: np.ndarray, escore: np.ndarray, probabilidade: np.ndarray, repeticoes: int = 300, semente: int = SEMENTE, minimo: int = 100
) -> pd.DataFrame:
    """Onde o alerta erra mais: AUC, reclamações perdidas e alarmes falsos em cada recorte do teste.

    Reclamação perdida = reclamação com probabilidade abaixo de 50%; alarme falso = não reclamação
    com probabilidade de 50% ou mais. Os recortes demográficos são agregados (nunca por pessoa).
    """
    recortes = {
        "tamanho do texto": faixa_de_tamanho(teste["palavras"]),
        "assunto": np.where(teste["texto"].str.contains(TERMOS_DE_ENTREGA, regex=True), "fala de entrega ou prazo", "não fala de entrega"),
        "gênero": teste["genero"].to_numpy(),
        "região": teste["regiao"].to_numpy(),
        "idade": faixa_de_idade(teste["nascimento"]),
    }
    marcado = probabilidade >= 0.5
    linhas = []
    for recorte, valores in recortes.items():
        valores = np.asarray(valores)
        for valor in pd.unique(valores):
            dentro = valores == valor
            if dentro.sum() < minimo or y[dentro].all() or not y[dentro].any():
                continue
            inferior, superior = metricas.bootstrap(metricas.auc, y[dentro], escore[dentro], repeticoes=repeticoes, semente=semente)
            linhas.append(
                {
                    "recorte": recorte, "grupo": str(valor), "avaliações": int(dentro.sum()), "reclamações": float(y[dentro].mean()),
                    "AUC": metricas.auc(y[dentro], escore[dentro]), "IC 95% inferior": inferior, "IC 95% superior": superior,
                    "reclamações perdidas": float((~marcado[dentro & y]).mean()),
                    "alarmes falsos": float(marcado[dentro & ~y].mean()),
                }
            )
    return pd.DataFrame(linhas).sort_values(["recorte", "grupo"]).set_index(["recorte", "grupo"])


def leitura_manual(caminho) -> pd.DataFrame:
    """Resumo da leitura manual dos erros mais confiantes do modelo (arquivo versionado, sem texto).

    Foram lidos os 45 alarmes falsos de maior escore (nota 4 ou 5 e "recomendaria") e as 45
    reclamações perdidas de menor escore (nota até 2 e "não recomendaria") do teste, e cada um
    recebeu uma categoria. A tabela traz a contagem por categoria com intervalo de Wilson.
    """
    leitura = pd.read_csv(caminho)
    linhas = []
    for tipo, grupo in leitura.groupby("tipo"):
        for categoria, quantos in grupo["categoria"].value_counts().items():
            inferior, superior = metricas.intervalo_de_wilson(int(quantos), len(grupo))
            linhas.append(
                {"tipo": tipo, "categoria": categoria, "avaliações": int(quantos), "de": len(grupo), "parcela": quantos / len(grupo), "IC 95% inferior": inferior, "IC 95% superior": superior}
            )
    return pd.DataFrame(linhas).set_index(["tipo", "categoria"])


def leitura_no_modelo_final(caminho, teste: pd.DataFrame, escore: np.ndarray) -> dict:
    """Quantas das avaliações lidas continuam entre os erros mais confiantes do modelo final.

    A leitura foi feita com o escore de uma versão preliminar do modelo. Aqui a seleção é refeita
    com o escore final (os alarmes falsos de maior escore entre nota 4 ou 5 e "recomendaria"; as
    reclamações perdidas de menor escore entre nota até 2 e "não recomendaria") e comparada.
    """
    leitura = pd.read_csv(caminho)
    ordenado = teste.assign(escore=escore)
    candidatos = {
        "alarme falso": ordenado[(ordenado["nota"] >= 4) & ordenado["recomenda"]].sort_values("escore", ascending=False),
        "reclamação perdida": ordenado[(ordenado["nota"] <= 2) & ~ordenado["recomenda"]].sort_values("escore", ascending=True),
    }
    resumo = {}
    for tipo, tabela in candidatos.items():
        lidas = set(leitura.loc[leitura["tipo"] == tipo, "chave"])
        resumo[tipo] = {"lidas": len(lidas), "ainda_entre_as_mais_extremas": len(lidas & set(tabela["chave"].head(len(lidas))))}
    return resumo


# ---------------------------------------------------------------------------------------------
# Motivos de reclamação
# ---------------------------------------------------------------------------------------------

# Motivos e palavras-âncora, escritos à mão depois de ler as palavras dos temas no período de treino.
# A regra de `temas.nomear` aplica este dicionário a qualquer execução do NMF ou do k-means.
MOTIVOS: dict[str, list[str]] = {
    "Produto não recebido": ["nao_recebi", "nao_chegou", "nao_entregue", "nao_avaliar", "nao_posso", "avaliar", "aguardando", "aguardo", "nao_entregaram", "chegar", "esperando"],
    "Atraso e prazo de entrega": ["entrega", "prazo", "atraso", "demora", "previsao", "atrasada", "atrasado", "dias", "uteis"],
    "Defeito, avaria ou item errado": ["defeito", "quebrado", "troca", "trocar", "devolucao", "faltando", "veio", "errado", "pecas"],
    "Desempenho ruim no uso": ["uso", "aparelho", "celular", "nao_funciona", "bateria", "funcionar", "parou", "meses", "liga", "tela"],
    "Material e acabamento ruins": ["qualidade", "material", "acabamento", "fragil", "baixa", "pessima", "plastico", "tecido", "ma"],
    "Diferente do esperado ou do anunciado": ["nao_gostei", "diferente", "foto", "propaganda", "enganosa", "anuncio", "tamanho", "cor", "pequeno", "esperava", "achei", "imagem", "descricao"],
    "Loja e atendimento": ["americanas", "loja", "site", "contato", "atendimento", "empresa", "cliente", "lojas", "compra", "cancelamento"],
    "Desabafo sem motivo claro": ["pessimo", "horrivel", "nao_recomendo", "nao_comprem", "dinheiro", "jogado", "lixo", "porcaria", "ninguem"],
}
OUTROS = "Outros"
FAMILIAS = {
    "Produto não recebido": "entrega e loja", "Atraso e prazo de entrega": "entrega e loja", "Loja e atendimento": "entrega e loja",
    "Defeito, avaria ou item errado": "produto", "Desempenho ruim no uso": "produto", "Material e acabamento ruins": "produto",
    "Diferente do esperado ou do anunciado": "produto", "Desabafo sem motivo claro": "sem motivo claro", OUTROS: "sem motivo claro",
}
NUMEROS_DE_TEMAS = (4, 6, 8, 10, 12)
# Escolhido por leitura, no período de treino: é o menor número (entre os testados) em que os oito
# motivos aparecem separados no modelo de referência. A estabilidade não ajuda a escolher: fica
# parecida com qualquer número de temas (ver `estabilidade_por_numero_de_temas`).
NUMERO_DE_TEMAS = 10
PALAVRAS_PARA_NOMEAR = 12


def reclamacao_clara(tabela: pd.DataFrame) -> np.ndarray:
    """Reclamação sem ambiguidade de rótulo: nota até 2 e "não recomendaria"."""
    return ((tabela["nota"] <= 2) & ~tabela["recomenda"]).to_numpy()


def vetorizador_de_temas() -> texto.TfidfDePalavras:
    """TF-IDF usado nos temas: palavras com a negação colada, sem palavras vazias, sem termos raros (menos de 10
    reclamações) nem onipresentes (mais de 35% das reclamações, como "produto")."""
    return texto.TfidfDePalavras(df_minimo=10, modo="negacao", ignorar=texto.PALAVRAS_VAZIAS, df_maximo=0.35)


def estabilidade_por_numero_de_temas(
    matriz, numeros: tuple[int, ...] = NUMEROS_DE_TEMAS, repeticoes: int = 5, semente: int = SEMENTE
) -> pd.DataFrame:
    """Para cada número de temas: estabilidade média, mínima e máxima entre pares de execuções do NMF."""
    linhas = []
    for k in numeros:
        media, minimo, maximo = temas.estabilidade_media(temas.execucoes_de_nmf(matriz, k, repeticoes, semente))
        linhas.append({"temas": k, "estabilidade": media, "mínimo": minimo, "máximo": maximo, "pares de execuções": repeticoes * (repeticoes - 1) // 2})
    return pd.DataFrame(linhas).set_index("temas")


def lista_de_motivos() -> list[str]:
    return [*MOTIVOS, OUTROS]


def motivo_de_cada_tema(pesos: np.ndarray, nomes: np.ndarray) -> list[str]:
    return temas.nomear(temas.palavras_do_topo(pesos, nomes, PALAVRAS_PARA_NOMEAR), MOTIVOS, sem_nome=OUTROS)


def motivo_de_cada_avaliacao(matriz, pesos: np.ndarray, nomes: np.ndarray) -> np.ndarray:
    """Índice do motivo dominante de cada avaliação, na ordem de `lista_de_motivos()` (sem tema = Outros)."""
    motivos = lista_de_motivos()
    do_tema = np.array([motivos.index(m) for m in motivo_de_cada_tema(pesos, nomes)] + [motivos.index(OUTROS)])
    return do_tema[temas.tema_dominante(temas.mistura(matriz, pesos))]  # o índice -1 cai na última posição: Outros


def descrever_temas(referencia: np.ndarray, execucoes: list[np.ndarray], nomes: np.ndarray, matriz_fora) -> pd.DataFrame:
    """Uma linha por tema do modelo de referência: motivo, palavras do topo, estabilidade e parcela das reclamações."""
    dominante = temas.tema_dominante(temas.mistura(matriz_fora, referencia))
    return pd.DataFrame(
        {
            "motivo": motivo_de_cada_tema(referencia, nomes),
            "palavras": [", ".join(p) for p in temas.palavras_do_topo(referencia, nomes, 10)],
            "estabilidade": temas.estabilidade_por_tema(referencia, execucoes),
            "parcela": temas.parcelas(dominante, referencia.shape[0]),
        },
        index=pd.Index(range(1, referencia.shape[0] + 1), name="tema"),
    )


def _parcelas_dos_motivos(indices: np.ndarray) -> np.ndarray:
    return np.bincount(indices, minlength=len(lista_de_motivos())) / len(indices)


def motivos_nas_execucoes(matriz, execucoes: list[np.ndarray], nomes: np.ndarray) -> list[np.ndarray]:
    """O motivo de cada avaliação segundo cada execução do NMF (os temas de cada uma recebem nome pela mesma regra)."""
    return [motivo_de_cada_avaliacao(matriz, pesos, nomes) for pesos in execucoes]


def tamanho_dos_motivos(
    referencia: np.ndarray, execucoes: list[np.ndarray], nomes: np.ndarray, matriz_fora, repeticoes: int = REPETICOES, semente: int = SEMENTE,
    das_execucoes: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    """Parcela das reclamações em cada motivo, com dois intervalos de 95%.

    - amostragem: os temas ficam fixos (modelo de referência) e as reclamações são reamostradas;
    - reajuste: além disso, os temas são reaprendidos (cada execução com outra semente e outra
      subamostra do treino recebe os nomes pela mesma regra). É o intervalo honesto: inclui a
      incerteza de o próprio tema mudar.
    """
    motivos = lista_de_motivos()
    gerador = np.random.default_rng(semente)
    n = matriz_fora.shape[0]
    da_referencia = motivo_de_cada_avaliacao(matriz_fora, referencia, nomes)
    amostragem = np.array([_parcelas_dos_motivos(da_referencia[gerador.integers(0, n, size=n)]) for _ in range(repeticoes)])
    if das_execucoes is None:
        das_execucoes = motivos_nas_execucoes(matriz_fora, execucoes, nomes)
    por_execucao = max(1, repeticoes // len(execucoes))
    reajuste = np.array([_parcelas_dos_motivos(indices[gerador.integers(0, n, size=n)]) for indices in das_execucoes for _ in range(por_execucao)])
    temas_da_referencia = motivo_de_cada_tema(referencia, nomes)
    estabilidade = temas.estabilidade_por_tema(referencia, execucoes)
    tabela = pd.DataFrame(
        {
            "família": [FAMILIAS[m] for m in motivos],
            "parcela": _parcelas_dos_motivos(da_referencia),
            "amostragem inf": np.quantile(amostragem, 0.025, axis=0),
            "amostragem sup": np.quantile(amostragem, 0.975, axis=0),
            "reajuste inf": np.quantile(reajuste, 0.025, axis=0),
            "reajuste sup": np.quantile(reajuste, 0.975, axis=0),
            "menor entre execuções": np.min([_parcelas_dos_motivos(i) for i in das_execucoes], axis=0),
            "maior entre execuções": np.max([_parcelas_dos_motivos(i) for i in das_execucoes], axis=0),
            "temas": [temas_da_referencia.count(m) for m in motivos],
            "estabilidade": [
                float(np.mean([e for e, t in zip(estabilidade, temas_da_referencia) if t == m])) if m in temas_da_referencia else np.nan for m in motivos
            ],
            "execuções em que aparece": [sum(m in motivo_de_cada_tema(pesos, nomes) for pesos in execucoes) for m in motivos],
        },
        index=pd.Index(motivos, name="motivo"),
    )
    tabela.attrs["execucoes"] = len(execucoes)
    return tabela


def parcela_por_familia(
    indices: np.ndarray, das_execucoes: list[np.ndarray] | None = None, repeticoes: int = REPETICOES, semente: int = SEMENTE
) -> pd.DataFrame:
    """Parcela das reclamações por família de motivo (entrega e loja, produto, sem motivo claro).

    Traz o intervalo de amostragem do modelo de referência e, se vierem os motivos segundo cada
    execução, a menor e a maior parcela entre as execuções.
    """
    motivos = lista_de_motivos()
    familias = sorted(set(FAMILIAS.values()))
    familia_do_motivo = np.array([familias.index(FAMILIAS[m]) for m in motivos])

    def parcelas(f: np.ndarray) -> np.ndarray:
        return np.bincount(f, minlength=len(familias)) / len(f)

    da_avaliacao = familia_do_motivo[indices]
    inferior, superior = metricas.bootstrap(parcelas, da_avaliacao, repeticoes=repeticoes, semente=semente)
    tabela = pd.DataFrame({"parcela": parcelas(da_avaliacao), "IC 95% inferior": inferior, "IC 95% superior": superior}, index=pd.Index(familias, name="família"))
    if das_execucoes:
        entre_execucoes = np.array([parcelas(familia_do_motivo[i]) for i in das_execucoes])
        tabela["menor entre execuções"], tabela["maior entre execuções"] = entre_execucoes.min(axis=0), entre_execucoes.max(axis=0)
    return tabela


def motivos_por_grupo(indices: np.ndarray, grupo, maximo: int = 8, minimo: int = 200, ordenar: bool = True) -> pd.DataFrame:
    """Mistura de motivos dentro de cada grupo (categoria de produto ou mês): cada linha soma 1.

    Com `ordenar`, ficam os `maximo` grupos com mais reclamações, desde que tenham pelo menos
    `minimo`; sem, todos os grupos na ordem natural (é o caso dos meses).
    """
    motivos = lista_de_motivos()
    grupo = np.asarray(grupo)
    contagem = pd.Series(grupo).value_counts()
    escolhidos = [g for g in contagem.index[:maximo] if contagem[g] >= minimo] if ordenar else sorted(contagem.index)
    linhas = {g: np.bincount(indices[grupo == g], minlength=len(motivos)) / contagem[g] for g in escolhidos}
    tabela = pd.DataFrame.from_dict(linhas, orient="index", columns=motivos)
    tabela.insert(0, "reclamações", [int(contagem[g]) for g in escolhidos])
    return tabela


def comparar_agrupamentos(
    textos: list[str], densas: dict[str, np.ndarray], numero_de_temas: int, repeticoes: int = 6, semente: int = SEMENTE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NMF sobre TF-IDF contra k-means sobre cada representação densa, nas mesmas reclamações e no mesmo protocolo.

    Devolve (resumo, parcelas): o resumo traz a estabilidade de cada método, a semelhança das
    palavras dos seus grupos com os temas do NMF e a parcela das reclamações em que os dois
    métodos apontam o mesmo motivo; `parcelas` traz o tamanho de cada motivo segundo cada método.
    `densas` pode trazer a função que calcula a representação a partir do TF-IDF (caso do LSA).
    """
    vetorizador = vetorizador_de_temas()
    matriz = vetorizador.ajustar(textos)
    nomes = vetorizador.nomes
    motivos = lista_de_motivos()

    referencia = temas.ajustar_nmf(matriz, numero_de_temas, semente).components_
    do_nmf = motivo_de_cada_avaliacao(matriz, referencia, nomes)
    media, minimo, maximo = temas.estabilidade_media(temas.execucoes_de_nmf(matriz, numero_de_temas, repeticoes, semente))
    resumo = {"NMF sobre TF-IDF": {"estabilidade": media, "mínimo": minimo, "máximo": maximo, "palavras em comum com o NMF": 1.0, "mesmo motivo que o NMF": 1.0}}
    tamanhos = {"NMF sobre TF-IDF": _parcelas_dos_motivos(do_nmf)}
    for nome, vetores in densas.items():
        if callable(vetores):
            vetores = vetores(matriz)
        media, minimo, maximo = temas.estabilidade_media(temas.execucoes_de_kmeans(vetores, matriz, numero_de_temas, repeticoes, semente))
        grupos = KMeans(n_clusters=numero_de_temas, n_init=1, random_state=semente).fit(vetores).labels_
        pesos = temas.palavras_caracteristicas(matriz, grupos, numero_de_temas)
        do_kmeans = np.array([motivos.index(m) for m in motivo_de_cada_tema(pesos, nomes)])[grupos]
        resumo[nome] = {
            "estabilidade": media, "mínimo": minimo, "máximo": maximo,
            "palavras em comum com o NMF": temas.estabilidade(referencia, pesos),
            "mesmo motivo que o NMF": float((do_kmeans == do_nmf).mean()),
        }
        tamanhos[nome] = _parcelas_dos_motivos(do_kmeans)
    tabela = pd.DataFrame.from_dict(resumo, orient="index")
    tabela.attrs["reclamacoes"] = len(textos)
    return tabela, pd.DataFrame(tamanhos, index=pd.Index(motivos, name="motivo"))


# ---------------------------------------------------------------------------------------------
# Representação densa contra TF-IDF, na mesma amostra
# ---------------------------------------------------------------------------------------------


def comparar_representacoes(
    y_treino: np.ndarray, y_teste: np.ndarray, representacoes: dict[str, tuple], extras: dict[str, np.ndarray] | None = None,
    repeticoes: int = REPETICOES, semente: int = SEMENTE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Treina uma regressão logística por representação nas mesmas avaliações e compara no mesmo teste.

    `representacoes`: nome -> (matriz de treino, matriz de teste). A regularização de cada uma é
    escolhida numa parte reservada do treino (os últimos 20% do sorteio) e o modelo é reajustado
    no treino inteiro. `extras` são escores já prontos para o mesmo teste (por exemplo, o modelo
    treinado na base completa). Devolve o desempenho, as diferenças pareadas contra a primeira
    representação e o C escolhido de cada uma.
    """
    corte = int(0.8 * len(y_treino))
    escores, escolhidos = {}, {}
    for nome, (treino, teste) in representacoes.items():
        _, grade = modelos.escolher_regularizacao(treino[:corte], y_treino[:corte], treino[corte:], y_treino[corte:])
        escolhidos[nome] = float(grade["AUC na validação"].idxmax())
        escores[nome] = modelos.ajustar_logistica(treino, y_treino, escolhidos[nome]).decision_function(teste)
    escores.update(extras or {})
    primeira = next(iter(representacoes))
    pares = [(nome, primeira) for nome in escores if nome != primeira]
    tabela, comparacoes = desempenho(y_teste, escores, pares, repeticoes, semente)
    return tabela, comparacoes, escolhidos
