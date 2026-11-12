"""A análise inteira, na ordem em que o README conta: rótulos, alerta, calibração, erros e motivos.

`executar` devolve um objeto com todas as tabelas; scripts/rodar_analise.py grava o JSON e as
figuras, e o notebook mostra as mesmas tabelas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from nlp_avaliacoes import analise, dados, modelos, temas, texto

LEXICO = "Léxico de palavras negativas"
PALAVRAS = "TF-IDF de palavras + regressão logística"
CARACTERES = "TF-IDF de caracteres + regressão logística"
SO_A_NOTA = "Só a nota"
NOTA_MAIS_TEXTO = "Nota + texto"
AMOSTRA_TFIDF = "TF-IDF de palavras"
AMOSTRA_LSA = "LSA (200 dimensões)"
AMOSTRA_MINILM = "MiniLM multilíngue (384 dimensões)"
AMOSTRA_BASE_INTEIRA = "TF-IDF de palavras, treinado na base inteira"
KMEANS_MINILM = "k-means sobre MiniLM"
KMEANS_LSA = "k-means sobre LSA"
EXECUCOES_DO_REAJUSTE = 10


@dataclass
class Resultados:
    base: dict
    discordancia_por_nota: pd.DataFrame
    discordancia: dict
    regularizacao: dict[str, pd.DataFrame]
    principal: str
    melhor_na_validacao: str
    desempenho: pd.DataFrame
    diferencas: pd.DataFrame
    curvas: pd.DataFrame
    regra_da_nota: pd.DataFrame
    auc_por_rotulo: pd.DataFrame
    texto_dentro_da_nota: pd.DataFrame
    nota_e_texto: pd.DataFrame
    nota_e_texto_diferenca: pd.DataFrame
    nota_alta_texto_negativo: dict
    calibracao: dict
    curvas_de_calibracao: pd.DataFrame
    erros: pd.DataFrame
    leitura: pd.DataFrame
    leitura_no_modelo_final: dict
    estabilidade: pd.DataFrame
    numero_de_temas: int
    estabilidade_do_modelo: dict
    temas: pd.DataFrame
    motivos: pd.DataFrame
    familias: pd.DataFrame
    por_categoria: pd.DataFrame
    por_mes: pd.DataFrame
    amostra: dict
    representacoes: pd.DataFrame
    representacoes_diferencas: pd.DataFrame
    agrupamentos: pd.DataFrame
    agrupamentos_parcelas: pd.DataFrame
    densa_disponivel: bool
    tempos: dict = field(default_factory=dict)


def executar(tabela: pd.DataFrame, contagem: dict, repeticoes: int = analise.REPETICOES, falar: bool = True) -> Resultados:
    """Roda tudo. `repeticoes` é o número de reamostragens do bootstrap (menor só para testar o encadeamento)."""
    tempos: dict[str, float] = {}
    relogio = time.perf_counter()

    def marcar(etapa: str) -> None:
        nonlocal relogio
        agora = time.perf_counter()
        tempos[etapa] = round(agora - relogio, 1)
        if falar:
            print(f"  [{tempos[etapa]:6.1f} s] {etapa}", flush=True)
        relogio = agora

    # As tabelas que sustentam a conclusão (desempenho do alerta, curva de captura, tamanho dos motivos) usam
    # `repeticoes` reamostragens; as análises de apoio usam a metade, para a análise inteira caber em poucos minutos.
    secundarias = max(1, repeticoes // 2)

    treino, validacao, teste = (tabela[tabela["periodo"] == p] for p in ("treino", "validação", "teste"))
    y_treino, y_validacao, y_teste = (analise.rotulo(parte) for parte in (treino, validacao, teste))

    # 1. A base e os dois rótulos
    base = analise.resumo_da_base(tabela, contagem)
    base["reamostragens_do_bootstrap"] = {"tabelas principais": repeticoes, "análises de apoio": secundarias, "recortes da análise de erro": min(200, repeticoes)}
    discordancia_por_nota = analise.discordancia_por_nota(tabela)
    discordancia = analise.resumo_da_discordancia(tabela)
    marcar("base e rótulos")

    # 2. Representações esparsas: o vocabulário e o IDF só veem o treino
    de_palavras = texto.TfidfDePalavras(df_minimo=5)
    palavras = [de_palavras.ajustar(treino["texto"].tolist()), de_palavras.transformar(validacao["texto"].tolist()), de_palavras.transformar(teste["texto"].tolist())]
    marcar("TF-IDF de palavras")
    de_caracteres = texto.TfidfDeCaracteres(df_minimo=5)
    caracteres = [de_caracteres.ajustar(treino["texto"].tolist()), de_caracteres.transformar(validacao["texto"].tolist()), de_caracteres.transformar(teste["texto"].tolist())]
    base["termos_no_tfidf_de_palavras"] = int(palavras[0].shape[1])
    base["termos_no_tfidf_de_caracteres"] = int(caracteres[0].shape[1])
    marcar("TF-IDF de caracteres")

    # 3. Modelos: regularização escolhida na validação (abril)
    modelo_palavras, grade_palavras = modelos.escolher_regularizacao(palavras[0], y_treino, palavras[1], y_validacao)
    modelo_caracteres, grade_caracteres = modelos.escolher_regularizacao(caracteres[0], y_treino, caracteres[1], y_validacao)
    regularizacao = {PALAVRAS: grade_palavras, CARACTERES: grade_caracteres}
    escores = {
        LEXICO: texto.escore_do_lexico(teste["texto"]),
        PALAVRAS: modelo_palavras.decision_function(palavras[2]),
        CARACTERES: modelo_caracteres.decision_function(caracteres[2]),
    }
    base["parcela_do_teste_com_termo_do_lexico"] = float((escores[LEXICO] > 0).mean())
    del caracteres
    # O modelo principal do estudo é o de palavras (mais simples de explicar). O JSON registra qual dos dois teve a
    # maior AUC na validação, para o README poder dizer se a escolha coincide com a que os dados fariam.
    principal = PALAVRAS
    melhor_na_validacao = PALAVRAS if grade_palavras["AUC na validação"].max() >= grade_caracteres["AUC na validação"].max() else CARACTERES
    logito_validacao = modelo_palavras.decision_function(palavras[1])
    logito_teste = escores[principal]
    marcar("regressões logísticas")

    desempenho, diferencas = analise.desempenho(y_teste, escores, [(PALAVRAS, LEXICO), (PALAVRAS, CARACTERES)], repeticoes)
    curvas = analise.curvas_de_captura(y_teste, {PALAVRAS: escores[PALAVRAS], LEXICO: escores[LEXICO]}, com_faixa=PALAVRAS, repeticoes=repeticoes)
    regra_da_nota = analise.regra_da_nota(y_teste, teste["nota"].to_numpy(), logito_teste, repeticoes)
    marcar("desempenho do alerta com bootstrap")

    # 4. Ruído de rótulo
    c_escolhido = float(grade_palavras["AUC na validação"].idxmax())
    modelo_da_nota = modelos.ajustar_logistica(palavras[0], analise.rotulo(treino, "nota"), c_escolhido)
    auc_por_rotulo = analise.auc_por_rotulo(
        teste, {'"não recomendaria"': logito_teste, '"nota até 2"': modelo_da_nota.decision_function(palavras[2])}, secundarias
    )
    texto_dentro_da_nota = analise.texto_dentro_da_nota(teste, logito_teste, secundarias)
    de_segundo_estagio = analise.nota_mais_texto(validacao, logito_validacao, teste, logito_teste)
    nota_e_texto, nota_e_texto_diferenca = analise.desempenho(y_teste, de_segundo_estagio, [(NOTA_MAIS_TEXTO, SO_A_NOTA)], secundarias)
    marcar("ruído de rótulo")

    # 5. Calibração e erros
    calibracao, curvas_de_calibracao, probabilidade = analise.calibracao(y_validacao, logito_validacao, y_teste, logito_teste, secundarias)
    nota_alta_texto_negativo = analise.texto_negativo_com_nota_alta(teste, probabilidade)
    erros = analise.erros_por_recorte(teste, y_teste, logito_teste, probabilidade, repeticoes=min(200, repeticoes))
    leitura = analise.leitura_manual(dados.ARQUIVO_LEITURA)
    leitura_no_modelo_final = analise.leitura_no_modelo_final(dados.ARQUIVO_LEITURA, teste, logito_teste)
    marcar("calibração e análise de erro")

    # 6. Motivos de reclamação: temas aprendidos nas reclamações do treino, medidos depois
    clara = analise.reclamacao_clara(tabela)
    reclamacoes_do_treino = tabela[clara & (tabela["periodo"] == "treino").to_numpy()]
    reclamacoes_de_fora = tabela[clara & (tabela["periodo"] != "treino").to_numpy()]
    reclamacoes_do_teste = tabela[clara & (tabela["periodo"] == "teste").to_numpy()]
    de_temas = analise.vetorizador_de_temas()
    matriz_do_treino = de_temas.ajustar(reclamacoes_do_treino["texto"].tolist())
    matriz_de_fora = de_temas.transformar(reclamacoes_de_fora["texto"].tolist())
    matriz_do_teste = de_temas.transformar(reclamacoes_do_teste["texto"].tolist())
    nomes = de_temas.nomes
    numero_de_temas = analise.NUMERO_DE_TEMAS
    estabilidade = analise.estabilidade_por_numero_de_temas(matriz_do_treino, tuple(k for k in analise.NUMEROS_DE_TEMAS if k != numero_de_temas))
    marcar("estabilidade por número de temas")

    referencia = temas.ajustar_nmf(matriz_do_treino, numero_de_temas, analise.SEMENTE).components_
    execucoes = temas.execucoes_de_nmf(matriz_do_treino, numero_de_temas, EXECUCOES_DO_REAJUSTE, analise.SEMENTE + 1)
    media, minimo, maximo = temas.estabilidade_media(execucoes)
    estabilidade_do_modelo = {
        "temas": numero_de_temas, "execucoes": EXECUCOES_DO_REAJUSTE, "pares": EXECUCOES_DO_REAJUSTE * (EXECUCOES_DO_REAJUSTE - 1) // 2,
        "estabilidade": media, "minimo": minimo, "maximo": maximo,
        "palavras_repetidas_em_10": 20 * media / (1 + media),  # Jaccard J entre dois topos de 10 palavras = 20J/(1+J) palavras em comum
        "reclamacoes_no_treino": int(matriz_do_treino.shape[0]), "reclamacoes_no_teste": int(matriz_do_teste.shape[0]),
        "reclamacoes_fora_do_treino": int(matriz_de_fora.shape[0]), "termos": int(matriz_do_treino.shape[1]),
    }
    # o número de temas adotado entra na curva com as suas 10 execuções (45 pares), as mesmas do intervalo de reajuste
    estabilidade.loc[numero_de_temas] = [media, minimo, maximo, estabilidade_do_modelo["pares"]]
    estabilidade = estabilidade.sort_index().astype({"pares de execuções": int})
    tabela_de_temas = analise.descrever_temas(referencia, execucoes, nomes, matriz_do_teste)
    das_execucoes = analise.motivos_nas_execucoes(matriz_do_teste, execucoes, nomes)
    motivos = analise.tamanho_dos_motivos(referencia, execucoes, nomes, matriz_do_teste, repeticoes, das_execucoes=das_execucoes)
    do_teste = analise.motivo_de_cada_avaliacao(matriz_do_teste, referencia, nomes)
    familias = analise.parcela_por_familia(do_teste, das_execucoes, repeticoes)
    de_fora = analise.motivo_de_cada_avaliacao(matriz_de_fora, referencia, nomes)
    por_categoria = analise.motivos_por_grupo(de_fora, reclamacoes_de_fora["categoria"])
    todas = tabela[clara]
    de_todas = analise.motivo_de_cada_avaliacao(de_temas.transformar(todas["texto"].tolist()), referencia, nomes)
    por_mes = analise.motivos_por_grupo(de_todas, todas["data"].dt.strftime("%Y-%m"), ordenar=False)
    marcar("temas de referência e tamanho dos motivos")

    # 7. Representação densa, numa amostra fixa
    amostra = dados.amostra_para_densa(tabela)
    vetores = dados.carregar_embeddings(amostra["chave"])
    if vetores is None and falar:
        print("  AVISO: data/brutos/embeddings_minilm.npz não encontrado (ou de outra amostra). A comparação com a")
        print("         representação densa roda só com o LSA. Para incluir o MiniLM: scripts/gerar_embeddings.py")
    do_treino, do_teste_da_amostra = (amostra["parte"] == "treino").to_numpy(), (amostra["parte"] == "teste").to_numpy()
    y_amostra = analise.rotulo(amostra)
    da_amostra = texto.TfidfDePalavras(df_minimo=3)
    esparsa_treino = da_amostra.ajustar(amostra["texto"][do_treino].tolist())
    esparsa_teste = da_amostra.transformar(amostra["texto"][do_teste_da_amostra].tolist())
    lsa_treino, lsa_teste = modelos.projetar_lsa(esparsa_treino, [esparsa_teste], 200, analise.SEMENTE)
    representacoes = {AMOSTRA_TFIDF: (esparsa_treino, esparsa_teste), AMOSTRA_LSA: (lsa_treino, lsa_teste)}
    if vetores is not None:
        representacoes[AMOSTRA_MINILM] = (vetores[do_treino], vetores[do_teste_da_amostra])
    da_base_inteira = pd.Series(logito_teste, index=teste.index).loc[amostra.index[do_teste_da_amostra]].to_numpy()
    tabela_de_representacoes, diferencas_de_representacoes, c_da_amostra = analise.comparar_representacoes(
        y_amostra[do_treino], y_amostra[do_teste_da_amostra], representacoes, {AMOSTRA_BASE_INTEIRA: da_base_inteira}, secundarias
    )
    marcar("classificadores na amostra (esparso x denso)")

    reclamacao_da_amostra = analise.reclamacao_clara(amostra) & ~do_teste_da_amostra
    densas: dict = {}
    if vetores is not None:
        densas[KMEANS_MINILM] = vetores[reclamacao_da_amostra]
    densas[KMEANS_LSA] = lambda matriz: modelos.projetar_lsa(matriz, [], 100, analise.SEMENTE)[0]
    agrupamentos, agrupamentos_parcelas = analise.comparar_agrupamentos(amostra["texto"][reclamacao_da_amostra].tolist(), densas, numero_de_temas)
    marcar("agrupamentos na amostra (NMF x k-means)")

    return Resultados(
        base=base, discordancia_por_nota=discordancia_por_nota, discordancia=discordancia, regularizacao=regularizacao, principal=principal,
        melhor_na_validacao=melhor_na_validacao,
        desempenho=desempenho, diferencas=diferencas, curvas=curvas, regra_da_nota=regra_da_nota, auc_por_rotulo=auc_por_rotulo,
        texto_dentro_da_nota=texto_dentro_da_nota, nota_e_texto=nota_e_texto, nota_e_texto_diferenca=nota_e_texto_diferenca,
        nota_alta_texto_negativo=nota_alta_texto_negativo, calibracao=calibracao, curvas_de_calibracao=curvas_de_calibracao, erros=erros,
        leitura=leitura, leitura_no_modelo_final=leitura_no_modelo_final, estabilidade=estabilidade, numero_de_temas=numero_de_temas, estabilidade_do_modelo=estabilidade_do_modelo, temas=tabela_de_temas, motivos=motivos, familias=familias,
        por_categoria=por_categoria, por_mes=por_mes,
        amostra={
            "avaliacoes": int(len(amostra)), "treino": int(do_treino.sum()), "teste": int(do_teste_da_amostra.sum()),
            "reclamacoes_para_agrupar": int(reclamacao_da_amostra.sum()), "regularizacao": c_da_amostra,
            "termos_do_tfidf": int(esparsa_treino.shape[1]),
        },
        representacoes=tabela_de_representacoes, representacoes_diferencas=diferencas_de_representacoes,
        agrupamentos=agrupamentos, agrupamentos_parcelas=agrupamentos_parcelas, densa_disponivel=vetores is not None, tempos=tempos,
    )
