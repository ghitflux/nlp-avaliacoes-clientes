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
        "graficos_extras": graficos_extras(r),
    }


# ---------------------------------------------------------------------------------------------
# Gráficos extras: o equivalente em dados das figuras do README que não são o gráfico principal
# ---------------------------------------------------------------------------------------------

GABARITOS = {
    "não recomendaria": {"pt": "Não recomendaria", "en": "Would not recommend", "es": "No recomendaría"},
    "nota até 2": {"pt": "Nota até 2", "en": "Rating of 2 or less", "es": "Nota de 2 o menos"},
    "só onde os dois concordam": {"pt": "Só onde os dois concordam", "en": "Only where both agree", "es": "Solo donde ambos coinciden"},
    "qualquer um dos dois sinais": {"pt": "Qualquer um dos dois", "en": "Either signal", "es": "Cualquiera de los dos"},
}
CATEGORIAS_TRADUZIDAS = {
    "Celulares e Smartphones": {"en": "Mobile phones", "es": "Celulares y smartphones"},
    "Móveis": {"en": "Furniture", "es": "Muebles"},
    "Informática e Acessórios": {"en": "Computers and accessories", "es": "Informática y accesorios"},
    "Beleza e Perfumaria": {"en": "Beauty and fragrance", "es": "Belleza y perfumería"},
    "Utilidades Domésticas": {"en": "Housewares", "es": "Artículos para el hogar"},
    "Eletroportáteis": {"en": "Small appliances", "es": "Pequeños electrodomésticos"},
    "Casa e Construção": {"en": "Home improvement", "es": "Casa y construcción"},
    "TV e Home Theater": {"en": "TV and home theater", "es": "TV y home theater"},
    "Eletrodomésticos": {"en": "Home appliances", "es": "Electrodomésticos"},
    "Brinquedos": {"en": "Toys", "es": "Juguetes"},
    "Informática": {"en": "Computers", "es": "Informática"},
    "Livros": {"en": "Books", "es": "Libros"},
}


def _traduzir_categoria(nome: str) -> dict:
    """Nome da categoria nos três idiomas (sem tradução cadastrada, fica o nome original)."""
    return {"pt": nome, **{i: CATEGORIAS_TRADUZIDAS.get(nome, {}).get(i, nome) for i in ("en", "es")}}


def graficos_extras(r: pipeline.Resultados) -> list[dict]:
    """Cinco gráficos, do achado mais importante para o menos: ruído de rótulo, tamanho dos motivos,
    estabilidade dos temas, calibração e mistura de motivos por categoria."""

    def p(valor: float, casas: int = 0) -> dict:
        return {i: _pct(valor, i, casas) for i in IDIOMAS}

    def n(valor: float, casas: int = 2) -> dict:
        return {i: _num(valor, i, casas) for i in IDIOMAS}

    def r4(valor: float) -> float:
        return round(float(valor), 4)

    extras = []

    # 1. AUC do mesmo modelo conforme o gabarito (figura auc_por_rotulo.png)
    por_rotulo = r.auc_por_rotulo
    gabaritos = list(GABARITOS)
    rotulos = {i: [GABARITOS[g][i] for g in gabaritos] for i in IDIOMAS}
    series = []
    for treinado, papel, nome in (
        ('"não recomendaria"', "principal", {"pt": 'Treinado com "não recomendaria"', "en": 'Trained on "would not recommend"', "es": 'Entrenado con "no recomendaría"'}),
        ('"nota até 2"', "alternativa", {"pt": 'Treinado com "nota até 2"', "en": 'Trained on "rating of 2 or less"', "es": 'Entrenado con "nota de 2 o menos"'}),
    ):
        series.append({"nome": nome, "papel": papel, "rotulos": rotulos, "pontos": [[rotulos["pt"][k], r4(por_rotulo.loc[(treinado, g), "AUC"])] for k, g in enumerate(gabaritos)]})
    menor, maior = n(por_rotulo.loc['"não recomendaria"', "AUC"].min(), 3), n(por_rotulo.loc['"não recomendaria"', "AUC"].max(), 3)
    discordam = p(r.discordancia["rotulos_discordam"]["parcela"], 1)
    extras.append(
        {
            "tipo": "barras",
            "titulo": {
                "pt": f"O mesmo modelo vale de {menor['pt']} a {maior['pt']} de AUC, conforme o gabarito",
                "en": f"The same model scores {menor['en']} to {maior['en']} AUC, depending on the answer key",
                "es": f"El mismo modelo vale de {menor['es']} a {maior['es']} de AUC, según la referencia",
            },
            "eixo_x": {"pt": "Rótulo usado como gabarito", "en": "Label used as answer key", "es": "Etiqueta usada como referencia"},
            "eixo_y": {"pt": "AUC no teste", "en": "Test AUC", "es": "AUC en la prueba"},
            "formato_x": "numero",
            "formato_y": "numero",
            "rotulos": rotulos,
            "series": series,
            "leitura": {
                "pt": f"A nota e o “recomendaria” discordam em {discordam['pt']} das avaliações; trocar o gabarito muda a métrica, trocar o rótulo de treino quase não muda o modelo.",
                "en": f"Star rating and “would recommend” disagree on {discordam['en']} of reviews; swapping the answer key moves the metric, while swapping the training label barely changes the model.",
                "es": f"La nota y el “recomendaría” no coinciden en el {discordam['es']} de las reseñas; cambiar la referencia mueve la métrica, cambiar la etiqueta de entrenamiento casi no cambia el modelo.",
            },
        }
    )

    # 2. Tamanho de cada motivo (figura motivos.png)
    motivos = r.motivos[(r.motivos["temas"] > 0) | (r.motivos["maior entre execuções"] > 0.005)].sort_values("parcela", ascending=False)
    nomes = list(motivos.index)
    rotulos = {i: [MOTIVOS_TRADUZIDOS[m][i] for m in nomes] for i in IDIOMAS}
    amplitude = motivos["reajuste sup"] - motivos["reajuste inf"]
    instavel = amplitude.idxmax()
    baixo, alto = p(motivos.loc[instavel, "reajuste inf"]), p(motivos.loc[instavel, "reajuste sup"])
    extras.append(
        {
            "tipo": "barras",
            "titulo": {
                "pt": f"O tamanho de cada motivo muda ao reaprender os temas: de {baixo['pt']} a {alto['pt']} no extremo",
                "en": f"Reason sizes shift when topics are re-learned: from {baixo['en']} to {alto['en']} at the extreme",
                "es": f"El tamaño de cada motivo cambia al reaprender los temas: del {baixo['es']} al {alto['es']}",
            },
            "eixo_x": {"pt": "Motivo de reclamação", "en": "Complaint reason", "es": "Motivo de queja"},
            "eixo_y": {"pt": "Parcela das reclamações", "en": "Share of complaints", "es": "Parte de las quejas"},
            "formato_x": "numero",
            "formato_y": "percentual",
            "rotulos": rotulos,
            "series": [
                {"nome": {"pt": "Mínimo ao reaprender os temas", "en": "Low end when re-learning topics", "es": "Mínimo al reaprender los temas"},
                 "papel": "neutro", "rotulos": rotulos, "pontos": [[rotulos["pt"][k], r4(motivos.loc[m, "reajuste inf"])] for k, m in enumerate(nomes)]},
                {"nome": {"pt": "Modelo de referência", "en": "Reference model", "es": "Modelo de referencia"},
                 "papel": "principal", "rotulos": rotulos, "pontos": [[rotulos["pt"][k], r4(motivos.loc[m, "parcela"])] for k, m in enumerate(nomes)]},
                {"nome": {"pt": "Máximo ao reaprender os temas", "en": "High end when re-learning topics", "es": "Máximo al reaprender los temas"},
                 "papel": "neutro", "rotulos": rotulos, "pontos": [[rotulos["pt"][k], r4(motivos.loc[m, "reajuste sup"])] for k, m in enumerate(nomes)]},
            ],
            "leitura": {
                "pt": "Parcela das reclamações de maio por motivo, com o intervalo de 95% obtido reaprendendo os temas: os motivos se repetem, o tamanho deles não; use como mapa, não como meta.",
                "en": "Share of May complaints by reason, with the 95% range from re-learning the topics: the reasons recur but their sizes don't, so treat them as a map, not a target.",
                "es": "Parte de las quejas de mayo por motivo, con el intervalo del 95 % al reaprender los temas: los motivos se repiten, su tamaño no; úselos como mapa, no como meta.",
            },
        }
    )

    # 3. Estabilidade dos temas por número de temas (figura estabilidade_dos_temas.png)
    estabilidade = r.estabilidade
    x = [int(k) for k in estabilidade.index]
    series = [
        {"nome": {"pt": "NMF sobre TF-IDF", "en": "NMF on TF-IDF", "es": "NMF sobre TF-IDF"}, "papel": "principal",
         "pontos": [[k, r4(v)] for k, v in zip(x, estabilidade["estabilidade"])]},
    ]
    nomes_kmeans = {
        pipeline.KMEANS_MINILM: {"pt": "k-means sobre MiniLM (amostra)", "en": "k-means on MiniLM (sample)", "es": "k-means sobre MiniLM (muestra)"},
        pipeline.KMEANS_LSA: {"pt": "k-means sobre LSA (amostra)", "en": "k-means on LSA (sample)", "es": "k-means sobre LSA (muestra)"},
    }
    for (nome, valor), papel in zip(estabilidade_dos_agrupamentos(r).items(), ("alternativa", "neutro")):
        series.append({"nome": nomes_kmeans[nome], "papel": papel, "pontos": [[int(r.numero_de_temas), r4(valor)]]})
    lo, hi = n(estabilidade["estabilidade"].min()), n(estabilidade["estabilidade"].max())
    adotada = n(r.estabilidade_do_modelo["estabilidade"])
    extras.append(
        {
            "tipo": "linhas",
            "titulo": {
                "pt": f"Os temas mudam a cada execução: estabilidade de {lo['pt']} a {hi['pt']}",
                "en": f"Topics change from run to run: stability between {lo['en']} and {hi['en']}",
                "es": f"Los temas cambian en cada ejecución: estabilidad de {lo['es']} a {hi['es']}",
            },
            "eixo_x": {"pt": "Número de temas", "en": "Number of topics", "es": "Número de temas"},
            "eixo_y": {"pt": "Estabilidade (1 = mesmos temas sempre)", "en": "Stability (1 = same topics every time)", "es": "Estabilidad (1 = mismos temas siempre)"},
            "formato_x": "numero",
            "formato_y": "numero",
            "series": series,
            "faixa": {
                "serie": 0,
                "inferior": [[k, r4(v)] for k, v in zip(x, estabilidade["mínimo"])],
                "superior": [[k, r4(v)] for k, v in zip(x, estabilidade["máximo"])],
            },
            "leitura": {
                "pt": f"Quanto das palavras de cada tema se repete entre duas execuções (outra semente e outra subamostra); com os {r.numero_de_temas} temas adotados, {adotada['pt']}, e o agrupamento sobre a representação densa é menos estável.",
                "en": f"How much of each topic's vocabulary repeats between two runs (new seed, new subsample); with the {r.numero_de_temas} topics adopted it is {adotada['en']}, and clustering dense embeddings is less stable.",
                "es": f"Cuánto del vocabulario de cada tema se repite entre dos ejecuciones (otra semilla y otra submuestra); con los {r.numero_de_temas} temas adoptados es {adotada['es']}, y agrupar la representación densa es menos estable.",
            },
        }
    )

    # 4. Calibração (figura calibracao.png)
    curvas = r.curvas_de_calibracao
    bruta, recalibrada = curvas[curvas["versão"] == "bruta"], curvas[curvas["versão"] == "recalibrada"]
    erro_b, erro_r = n(r.calibracao["bruta"]["erro_de_calibracao"], 3), n(r.calibracao["recalibrada"]["erro_de_calibracao"], 3)
    extras.append(
        {
            "tipo": "linhas",
            "titulo": {
                "pt": f"Recalibrar reduz o erro de calibração de {erro_b['pt']} para {erro_r['pt']}",
                "en": f"Recalibration cuts the calibration error from {erro_b['en']} to {erro_r['en']}",
                "es": f"Recalibrar reduce el error de calibración de {erro_b['es']} a {erro_r['es']}",
            },
            "eixo_x": {"pt": "Probabilidade prevista de reclamação", "en": "Predicted probability of complaint", "es": "Probabilidad prevista de queja"},
            "eixo_y": {"pt": "Parcela que era reclamação", "en": "Share that were complaints", "es": "Parte que eran quejas"},
            "formato_x": "percentual",
            "formato_y": "percentual",
            "series": [
                {"nome": {"pt": "Recalibrada (Platt, ajustada em abril)", "en": "Recalibrated (Platt, fitted on April)", "es": "Recalibrada (Platt, ajustada en abril)"},
                 "papel": "principal", "pontos": [[r4(a_), r4(b_)] for a_, b_ in zip(recalibrada["prevista"], recalibrada["observada"])]},
                {"nome": {"pt": "Probabilidade bruta do modelo", "en": "Raw model probability", "es": "Probabilidad bruta del modelo"},
                 "papel": "base", "pontos": [[r4(a_), r4(b_)] for a_, b_ in zip(bruta["prevista"], bruta["observada"])]},
                {"nome": {"pt": "Calibração perfeita", "en": "Perfect calibration", "es": "Calibración perfecta"}, "papel": "neutro", "pontos": [[0.0, 0.0], [1.0, 1.0]]},
            ],
            "leitura": {
                "pt": "Cada ponto é um décimo das avaliações de maio: quanto mais perto da diagonal, mais a probabilidade prometida se cumpre; a recalibração melhora, mas não zera o erro.",
                "en": "Each point is a tenth of the May reviews: the closer to the diagonal, the more the promised probability comes true; recalibration helps but does not remove the error.",
                "es": "Cada punto es una décima parte de las reseñas de mayo: cuanto más cerca de la diagonal, más se cumple la probabilidad prometida; recalibrar mejora, pero no elimina el error.",
            },
        }
    )

    # 5. Mistura de motivos por categoria (figura motivos_por_categoria.png, um mapa de calor; aqui, os 4 maiores motivos)
    por_categoria = r.por_categoria
    quatro = list(r.motivos["parcela"].drop(index=OUTROS_NOME).sort_values(ascending=False).index[:4])
    categorias = list(por_categoria.index)
    rotulos = {i: [_traduzir_categoria(c)[i] for c in categorias] for i in IDIOMAS}
    papeis = ("principal", "base", "alternativa", "neutro")
    series = [
        {"nome": MOTIVOS_TRADUZIDOS[m], "papel": papel, "rotulos": rotulos, "pontos": [[rotulos["pt"][k], r4(por_categoria.loc[c, m])] for k, c in enumerate(categorias)]}
        for m, papel in zip(quatro, papeis)
    ]
    maior = MOTIVOS_TRADUZIDOS[quatro[0]]
    do_maior = por_categoria[quatro[0]]
    piso, teto = p(do_maior.min()), p(do_maior.max())
    extras.append(
        {
            "tipo": "barras",
            "titulo": {
                "pt": f"“{maior['pt']}” vai de {piso['pt']} a {teto['pt']} das reclamações, conforme o produto",
                "en": f"“{maior['en']}” ranges from {piso['en']} to {teto['en']} of complaints by product",
                "es": f"“{maior['es']}” va del {piso['es']} al {teto['es']} de las quejas según el producto",
            },
            "eixo_x": {"pt": "Categoria de produto", "en": "Product category", "es": "Categoría de producto"},
            "eixo_y": {"pt": "Parcela das reclamações da categoria", "en": "Share of the category's complaints", "es": "Parte de las quejas de la categoría"},
            "formato_x": "numero",
            "formato_y": "percentual",
            "rotulos": rotulos,
            "series": series,
            "leitura": {
                "pt": "Os quatro maiores motivos dentro de cada categoria (abril e maio): móvel reclama mais de logística, eletrônico mais de desempenho; só diferenças grandes superam a incerteza dos temas.",
                "en": "The four largest reasons within each category (April and May): furniture complaints lean toward logistics, electronics toward performance; only large gaps exceed the topic uncertainty.",
                "es": "Los cuatro motivos principales dentro de cada categoría (abril y mayo): los muebles se quejan más de la logística y la electrónica del rendimiento; solo las diferencias grandes superan la incertidumbre de los temas.",
            },
        }
    )
    return extras


OUTROS_NOME = analise.OUTROS
TIPOS = ("linhas", "barras", "pontos")
PAPEIS = ("principal", "base", "alternativa", "neutro")


def conferir_grafico(grafico: dict, com_leitura: bool) -> None:
    """Forma de uma especificação de gráfico do site (tipos, limites, três idiomas)."""
    assert grafico["tipo"] in TIPOS
    campos = ["titulo", "eixo_x", "eixo_y"] + (["leitura"] if com_leitura else [])
    for campo in campos:
        assert campo in grafico, f"falta o campo {campo}"
        assert set(grafico[campo]) == set(IDIOMAS) and all(grafico[campo][i].strip() for i in IDIOMAS), campo
    if com_leitura:
        assert all(len(t) <= 90 for t in grafico["titulo"].values()), grafico["titulo"]
    assert 1 <= len(grafico["series"]) <= 4
    limite = 300 if grafico["tipo"] == "pontos" else 80
    for serie in grafico["series"]:
        assert serie["papel"] in PAPEIS and set(serie["nome"]) == set(IDIOMAS)
        assert 1 <= len(serie["pontos"]) <= limite
        if grafico["tipo"] == "barras":
            assert all(isinstance(rotulo, str) and np.isfinite(valor) for rotulo, valor in serie["pontos"])
            if "rotulos" in serie:
                assert set(serie["rotulos"]) == set(IDIOMAS)
                assert all(len(serie["rotulos"][i]) == len(serie["pontos"]) for i in IDIOMAS)
                assert [rot for rot, _ in serie["pontos"]] == serie["rotulos"]["pt"]
        else:
            assert all(len(ponto) == 2 and all(np.isfinite(v) for v in ponto) for ponto in serie["pontos"])
    if "faixa" in grafico:
        assert grafico["tipo"] == "linhas" and 0 <= grafico["faixa"]["serie"] < len(grafico["series"])
        assert len(grafico["faixa"]["inferior"]) == len(grafico["faixa"]["superior"])


def conferir(bloco: dict) -> None:
    """Regras do bloco do site: número-chave curto, três idiomas em tudo, até 4 séries e 80 pontos por série."""
    for campo in ("numero_chave", "pergunta", "resultado", "detalhe", "dados"):
        assert set(bloco[campo]) == set(IDIOMAS), campo
    assert all(len(texto) <= 12 for texto in bloco["numero_chave"].values())
    grafico = bloco["grafico"]
    assert grafico["tipo"] in ("linhas", "barras") and len(grafico["series"]) <= 4
    assert all(len(serie["pontos"]) <= 80 and serie["papel"] in ("principal", "base", "alternativa", "neutro") for serie in grafico["series"])
    assert all(np.isfinite(valor) for serie in grafico["series"] for ponto in serie["pontos"] for valor in ponto)
    extras = bloco["graficos_extras"]
    assert 3 <= len(extras) <= 5
    for extra in extras:
        conferir_grafico(extra, com_leitura=True)
