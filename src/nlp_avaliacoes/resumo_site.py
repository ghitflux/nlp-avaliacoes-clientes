"""Números-chave do README (`achados`) e o bloco trilíngue `para_o_site` de reports/resultados.json.

Todo número dos textos vem das tabelas da análise; nada é digitado à mão.
"""

from __future__ import annotations

import numpy as np

from nlp_avaliacoes import analise, pipeline, texto

IDIOMAS = ("pt", "en", "es")
MOTIVOS_TRADUZIDOS = {
    "Produto não recebido": {"pt": "Produto não recebido", "en": "Product never arrived", "es": "Producto no recibido"},
    "Atraso e prazo de entrega": {"pt": "Atraso e prazo de entrega", "en": "Late delivery", "es": "Retraso en la entrega"},
    "Defeito, avaria ou item errado": {"pt": "Defeito, avaria ou item errado", "en": "Defective, damaged or wrong item", "es": "Defecto, daño o artículo equivocado"},
    "Desempenho ruim no uso": {"pt": "Desempenho ruim no uso", "en": "Poor performance in use", "es": "Mal rendimiento en el uso"},
    "Material e acabamento ruins": {"pt": "Material e acabamento ruins", "en": "Poor material and finish", "es": "Material y acabado deficientes"},
    "Diferente do esperado ou do anunciado": {"pt": "Diferente do esperado ou do anunciado", "en": "Not as expected or advertised", "es": "Distinto de lo esperado o anunciado"},
    "Loja e atendimento": {"pt": "Loja e atendimento", "en": "Store and customer service", "es": "Tienda y atención al cliente"},
    "Desabafo sem motivo claro": {"pt": "Desabafo sem motivo claro", "en": "Venting with no clear reason", "es": "Desahogo sin motivo claro"},
    "Outros": {"pt": "Outros", "en": "Other", "es": "Otros"},
}


def _pct(valor: float, idioma: str, casas: int = 0) -> str:
    """Percentual no formato do idioma (em espanhol, com espaço antes do símbolo)."""
    texto = f"{valor * 100:.{casas}f}"
    if idioma != "en":
        texto = texto.replace(".", ",")
    return f"{texto} %" if idioma == "es" else f"{texto}%"


def _num(valor: float, idioma: str, casas: int = 2) -> str:
    texto = f"{valor:.{casas}f}"
    return texto if idioma == "en" else texto.replace(".", ",")


def _milhares(valor: int, idioma: str) -> str:
    """Número arredondado em milhares: "126 mil", "126,000", "126 000"."""
    mil = round(valor / 1000)
    return {"pt": f"{mil} mil", "en": f"{mil},000", "es": f"{mil} 000"}[idioma]


def estabilidade_dos_agrupamentos(r: pipeline.Resultados) -> dict[str, float]:
    """Estabilidade do k-means sobre cada representação densa disponível (para a figura de estabilidade)."""
    return {nome: float(r.agrupamentos.loc[nome, "estabilidade"]) for nome in (pipeline.KMEANS_MINILM, pipeline.KMEANS_LSA) if nome in r.agrupamentos.index}


def achados(r: pipeline.Resultados) -> dict:
    """Os números que o README cita e que não estão prontos numa célula de tabela."""
    modelo, lexico = r.desempenho.loc[pipeline.PALAVRAS], r.desempenho.loc[pipeline.LEXICO]
    prevalencia = r.base["periodos"]["teste"]["nao_recomendaria"]
    motivos = r.motivos[r.motivos["parcela"] > 0]
    amplitude = (r.motivos["reajuste sup"] - r.motivos["reajuste inf"]).drop(index="Outros")
    mais_instavel = amplitude.idxmax()
    alarmes = r.leitura.loc["alarme falso"]
    reais = int(alarmes.loc[["reclamação de entrega ou atendimento", "reclamação do produto recebido"], "avaliações"].sum())
    por_rotulo = r.auc_por_rotulo.loc['"não recomendaria"']
    categorias = r.por_categoria.drop(columns="reclamações")
    desvio = (categorias - r.motivos["parcela"]).abs()
    categoria, motivo = desvio.stack().idxmax()
    densa = pipeline.AMOSTRA_MINILM if r.densa_disponivel else pipeline.AMOSTRA_LSA
    return {
        "termos_no_lexico": len(texto.LEXICO_NEGATIVO),
        "parcela_do_teste_com_termo_do_lexico": r.base["parcela_do_teste_com_termo_do_lexico"],
        "prevalencia_no_teste": prevalencia,
        "teto_de_revocacao_lendo_10": min(1.0, analise.PARCELA_LIDA / prevalencia),
        "piso_de_leitura_para_80": analise.ALVO_DE_CAPTURA * prevalencia,
        "auc_do_modelo": modelo["AUC"], "auc_do_lexico": lexico["AUC"],
        "precisao_lendo_10_modelo": modelo["precisão lendo 10%"], "precisao_lendo_10_lexico": lexico["precisão lendo 10%"],
        "leitura_para_80_modelo": modelo["leitura para pegar 80%"], "leitura_para_80_lexico": lexico["leitura para pegar 80%"],
        "auc_menor_entre_rotulos": float(por_rotulo["AUC"].min()), "auc_maior_entre_rotulos": float(por_rotulo["AUC"].max()),
        "alarmes_falsos_lidos": int(alarmes["de"].iloc[0]), "alarmes_falsos_que_eram_reclamacao": reais,
        "estabilidade_dos_temas": r.estabilidade_do_modelo["estabilidade"],
        "estabilidade_menor_por_numero_de_temas": float(r.estabilidade["estabilidade"].min()),
        "estabilidade_maior_por_numero_de_temas": float(r.estabilidade["estabilidade"].max()),
        "motivos_no_modelo_de_referencia": int((r.motivos["temas"] > 0).sum()),
        "motivos_nomeados": len(analise.MOTIVOS),
        "execucoes_do_reajuste": int(r.motivos.attrs["execucoes"]),
        "motivos_em_todas_as_execucoes": int((r.motivos["execuções em que aparece"].drop(index="Outros") == r.motivos.attrs["execucoes"]).sum()),
        "maior_motivo": motivos["parcela"].idxmax(), "maior_motivo_parcela": float(motivos["parcela"].max()),
        "motivo_mais_instavel": mais_instavel,
        "motivo_mais_instavel_menor": float(r.motivos.loc[mais_instavel, "reajuste inf"]),
        "motivo_mais_instavel_maior": float(r.motivos.loc[mais_instavel, "reajuste sup"]),
        "motivo_mais_instavel_referencia": float(r.motivos.loc[mais_instavel, "parcela"]),
        "categoria_mais_diferente": categoria, "motivo_da_categoria_mais_diferente": motivo,
        "parcela_na_categoria": float(categorias.loc[categoria, motivo]), "parcela_geral_do_motivo": float(r.motivos.loc[motivo, "parcela"]),
        "representacao_densa_comparada": densa,
        "auc_da_densa_na_amostra": float(r.representacoes.loc[densa, "AUC"]),
        "auc_do_tfidf_na_amostra": float(r.representacoes.loc[pipeline.AMOSTRA_TFIDF, "AUC"]),
        "densa_empata_com_tfidf_em_auc": bool(r.representacoes_diferencas.loc[f"{densa} − {pipeline.AMOSTRA_TFIDF}", "AUC empata"]),
    }


def para_o_site(r: pipeline.Resultados) -> dict:
    a = achados(r)
    modelo, lexico = r.desempenho.loc[pipeline.PALAVRAS], r.desempenho.loc[pipeline.LEXICO]
    regra = r.regra_da_nota.loc['Regra "nota até 2"']
    texto_no_volume = r.regra_da_nota.loc["Texto, marcando o mesmo volume"]
    instavel = MOTIVOS_TRADUZIDOS[a["motivo_mais_instavel"]]
    usados = r.base["avaliacoes_usadas"]
    sempre, total, execucoes = a["motivos_em_todas_as_execucoes"], a["motivos_nomeados"], a["execucoes_do_reajuste"]

    def p(valor: float, casas: int = 0) -> dict:
        return {i: _pct(valor, i, casas) for i in IDIOMAS}

    def n(valor: float, casas: int = 2) -> dict:
        return {i: _num(valor, i, casas) for i in IDIOMAS}

    auc, precisao, precisao_lexico = n(modelo["AUC"]), p(modelo["precisão lendo 10%"]), p(lexico["precisão lendo 10%"])
    menor, maior = p(a["motivo_mais_instavel_menor"]), p(a["motivo_mais_instavel_maior"])
    ler, ler_lexico = p(modelo["leitura para pegar 80%"]), p(lexico["leitura para pegar 80%"])
    auc_densa, auc_tfidf = n(a["auc_da_densa_na_amostra"], 3), n(a["auc_do_tfidf_na_amostra"], 3)
    p_regra, p_texto = p(regra["precisão"]), p(texto_no_volume["precisão"])
    if r.densa_disponivel:
        densa = {
            "pt": f"A representação densa de um modelo de linguagem (MiniLM) não superou o TF-IDF na mesma amostra (AUC de {auc_densa['pt']} contra {auc_tfidf['pt']})",
            "en": f"Dense embeddings from a language model (MiniLM) did not beat TF-IDF on the same sample (AUC {auc_densa['en']} versus {auc_tfidf['en']})",
            "es": f"La representación densa de un modelo de lenguaje (MiniLM) no superó al TF-IDF en la misma muestra (AUC de {auc_densa['es']} frente a {auc_tfidf['es']})",
        }
    else:
        densa = {
            "pt": f"A representação densa (análise semântica latente) não superou o TF-IDF na mesma amostra (AUC de {auc_densa['pt']} contra {auc_tfidf['pt']})",
            "en": f"The dense representation (latent semantic analysis) did not beat TF-IDF on the same sample (AUC {auc_densa['en']} versus {auc_tfidf['en']})",
            "es": f"La representación densa (análisis semántico latente) no superó al TF-IDF en la misma muestra (AUC de {auc_densa['es']} frente a {auc_tfidf['es']})",
        }

    curvas = r.curvas
    x = curvas["parcela lida"].to_numpy()

    def pontos(coluna: str) -> list[list[float]]:
        return [[round(float(a_), 4), round(float(b_), 4)] for a_, b_ in zip(x, curvas[coluna].to_numpy())]

    return {
        "numero_chave": {"pt": f"AUC {auc['pt']}", "en": f"AUC {auc['en']}", "es": f"AUC {auc['es']}"},
        "pergunta": {
            "pt": "Do que os clientes reclamam, e dá para confiar numa leitura automática das avaliações?",
            "en": "What do customers complain about, and can an automated reading of their reviews be trusted?",
            "es": "¿De qué se quejan los clientes y se puede confiar en una lectura automática de sus reseñas?",
        },
        "resultado": {
            "pt": (
                f"Para o alerta de insatisfação, sim: um modelo simples (TF-IDF com regressão logística) separa reclamação de elogio com AUC de "
                f"{auc['pt']}, e lendo só os 10% de avaliações de maior escore o time encontra {precisao['pt']} de reclamações (com uma lista de "
                f"palavras negativas, {precisao_lexico['pt']}). Para a lista de motivos, só em parte: {sempre} dos {total} motivos reaparecem em "
                f"todas as {execucoes} execuções, mas o tamanho de cada um muda (“{instavel['pt']}” vai de {menor['pt']} a {maior['pt']} das "
                f"reclamações quando os temas são reaprendidos com outra semente)."
            ),
            "en": (
                f"For the dissatisfaction alert, yes: a simple model (TF-IDF with logistic regression) separates complaints from praise with an AUC of "
                f"{auc['en']}, and when the team reads only the top-scoring 10% of reviews, {precisao['en']} of them are complaints (a hand-made list "
                f"of negative words gets {precisao_lexico['en']}). For the list of reasons, only partly: {sempre} of the {total} reasons come back "
                f"in all {execucoes} runs, but their sizes shift (“{instavel['en']}” ranges from {menor['en']} to {maior['en']} of complaints "
                f"when the topics are re-learned with a different random seed)."
            ),
            "es": (
                f"Para la alerta de insatisfacción, sí: un modelo sencillo (TF-IDF con regresión logística) separa las quejas de los elogios con un AUC "
                f"de {auc['es']}, y al leer solo el 10 % de reseñas con mayor puntuación, el {precisao['es']} son quejas (con una lista de palabras "
                f"negativas, el {precisao_lexico['es']}). Para la lista de motivos, solo en parte: {sempre} de los {total} motivos reaparecen en "
                f"las {execucoes} ejecuciones, pero su tamaño cambia (“{instavel['es']}” va del {menor['es']} al {maior['es']} de las quejas "
                f"cuando los temas se vuelven a aprender con otra semilla)."
            ),
        },
        "detalhe": {
            "pt": (
                f"O gráfico mostra quanto das reclamações o time captura conforme a parcela das avaliações que consegue ler: para pegar 80% delas, "
                f"o modelo pede a leitura de {ler['pt']} das avaliações e o léxico, de {ler_lexico['pt']}. {densa['pt']}, e a nota de 1 a 5, quando "
                f"existe, ainda é um alerta melhor que o texto (precisão de {p_regra['pt']} contra {p_texto['pt']} marcando o mesmo volume)."
            ),
            "en": (
                f"The chart shows how many complaints the team catches as it reads a larger share of the reviews: to catch 80% of them, the model "
                f"needs {ler['en']} of the reviews read and the word list needs {ler_lexico['en']}. {densa['en']}, and the 1-to-5 star rating, "
                f"when it exists, is still a better alert than the text ({p_regra['en']} precision versus {p_texto['en']} flagging the same volume)."
            ),
            "es": (
                f"El gráfico muestra cuántas quejas captura el equipo según la parte de las reseñas que logra leer: para detectar el 80 % de ellas, "
                f"el modelo exige leer el {ler['es']} de las reseñas y el léxico, el {ler_lexico['es']}. {densa['es']}, y la nota de 1 a 5, cuando "
                f"existe, sigue siendo mejor alerta que el texto (precisión del {p_regra['es']} frente al {p_texto['es']} marcando el mismo volumen)."
            ),
        },
        "dados": {
            "pt": f"B2W-Reviews01: {_milhares(usados, 'pt')} avaliações de produtos de um e-commerce brasileiro, de janeiro a maio de 2018 (treino até março, teste em maio).",
            "en": f"B2W-Reviews01: {_milhares(usados, 'en')} product reviews in Portuguese from a Brazilian e-commerce site, January to May 2018 (trained through March, tested on May).",
            "es": f"B2W-Reviews01: {_milhares(usados, 'es')} reseñas de productos de un comercio electrónico brasileño, de enero a mayo de 2018 (entrenamiento hasta marzo, prueba en mayo).",
        },
        "etiquetas": ["NLP em português", "TF-IDF", "ruído de rótulo", "NMF", "estabilidade de temas", "calibração"],
        "grafico": {
            "tipo": "linhas",
            "titulo": {
                "pt": f"Para pegar 80% das reclamações, o modelo pede a leitura de {ler['pt']} das avaliações; o léxico, de {ler_lexico['pt']}",
                "en": f"To catch 80% of complaints, the model needs {ler['en']} of reviews read; the word list needs {ler_lexico['en']}",
                "es": f"Para detectar el 80 % de las quejas, el modelo exige leer el {ler['es']} de las reseñas; el léxico, el {ler_lexico['es']}",
            },
            "eixo_x": {"pt": "Parcela das avaliações lidas", "en": "Share of reviews read", "es": "Parte de las reseñas leídas"},
            "eixo_y": {"pt": "Parcela das reclamações capturadas", "en": "Share of complaints caught", "es": "Parte de las quejas detectadas"},
            "formato_x": "percentual",
            "formato_y": "percentual",
            "series": [
                {
                    "nome": {"pt": "Modelo de texto (TF-IDF + regressão logística)", "en": "Text model (TF-IDF + logistic regression)", "es": "Modelo de texto (TF-IDF + regresión logística)"},
                    "papel": "principal",
                    "pontos": pontos(pipeline.PALAVRAS),
                },
                {
                    "nome": {"pt": "Léxico de palavras negativas", "en": "Hand-made list of negative words", "es": "Léxico de palabras negativas"},
                    "papel": "base",
                    "pontos": pontos(pipeline.LEXICO),
                },
                {"nome": {"pt": "Ler ao acaso", "en": "Reading at random", "es": "Leer al azar"}, "papel": "neutro", "pontos": pontos("acaso")},
            ],
            "faixa": {"serie": 0, "inferior": pontos(f"{pipeline.PALAVRAS} inf"), "superior": pontos(f"{pipeline.PALAVRAS} sup")},
        },
    }


def conferir(bloco: dict) -> None:
    """Regras do bloco do site: número-chave curto, três idiomas em tudo, até 4 séries e 80 pontos por série."""
    for campo in ("numero_chave", "pergunta", "resultado", "detalhe", "dados"):
        assert set(bloco[campo]) == set(IDIOMAS), campo
    assert all(len(texto) <= 12 for texto in bloco["numero_chave"].values())
    grafico = bloco["grafico"]
    assert grafico["tipo"] in ("linhas", "barras") and len(grafico["series"]) <= 4
    assert all(len(serie["pontos"]) <= 80 and serie["papel"] in ("principal", "base", "alternativa", "neutro") for serie in grafico["series"])
    assert all(np.isfinite(valor) for serie in grafico["series"] for ponto in serie["pontos"] for valor in ponto)
