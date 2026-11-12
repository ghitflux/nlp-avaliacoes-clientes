import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nlp_avaliacoes import analise, dados, metricas


def _avaliacoes(n: int = 3000, semente: int = 0) -> pd.DataFrame:
    """Avaliações sintéticas: a nota vem de uma insatisfação latente; o "recomendaria" discorda em parte."""
    gerador = np.random.default_rng(semente)
    insatisfacao = gerador.normal(size=n)
    nota = np.clip(np.round(3 - 1.3 * insatisfacao + gerador.normal(scale=0.4, size=n)), 1, 5).astype(int)
    nao_recomenda = insatisfacao + gerador.normal(scale=0.5, size=n) > 0.7
    return pd.DataFrame(
        {
            "nota": nota,
            "recomenda": ~nao_recomenda,
            "insatisfacao": insatisfacao,
            "palavras": gerador.integers(1, 80, size=n),
            "texto": np.where(gerador.random(n) < 0.4, "a entrega atrasou", "produto fraco"),
            "genero": np.where(gerador.random(n) < 0.5, "F", "M"),
            "regiao": "Sudeste",
            "nascimento": gerador.choice([1960.0, 1975.0, 1990.0, np.nan], size=n),
        }
    )


def test_rotulos():
    tabela = pd.DataFrame({"nota": [1, 2, 3, 5], "recomenda": [False, True, False, True]})
    assert analise.rotulo(tabela, "recomendaria").tolist() == [True, False, True, False]
    assert analise.rotulo(tabela, "nota").tolist() == [True, True, False, False]
    assert analise.reclamacao_clara(tabela).tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        analise.rotulo(tabela, "outro")


def test_discordancia_caso_contado_a_mao():
    # nota 1: 3 avaliações, 2 não recomendam; nota 3: 2, 1 não recomenda; nota 5: 5, 1 não recomenda
    tabela = pd.DataFrame({"nota": [1, 1, 1, 3, 3, 5, 5, 5, 5, 5], "recomenda": [False, False, True, False, True, False, True, True, True, True]})
    por_nota = analise.discordancia_por_nota(tabela)
    assert por_nota["avaliações"].tolist() == [3, 2, 5]
    assert por_nota["não recomendaria"].tolist() == pytest.approx([2 / 3, 0.5, 0.2])
    assert (por_nota["IC 95% inferior"] < por_nota["não recomendaria"]).all() and (por_nota["não recomendaria"] < por_nota["IC 95% superior"]).all()
    resumo = analise.resumo_da_discordancia(tabela)
    assert resumo["nota_baixa_mas_recomendaria"]["avaliacoes"] == 1 and resumo["nota_baixa_mas_recomendaria"]["de"] == 3
    assert resumo["nota_alta_mas_nao_recomendaria"]["parcela"] == pytest.approx(0.2)
    assert resumo["nota_3_e_nao_recomendaria"]["parcela"] == pytest.approx(0.5)
    # discordam: a nota 1 que recomenda, a nota 3 que não recomenda e a nota 5 que não recomenda
    assert resumo["rotulos_discordam"]["avaliacoes"] == 3
    assert resumo["nao_recomendaria_com_nota_3_ou_mais"]["parcela"] == pytest.approx(2 / 4)


def test_desempenho_traz_intervalo_em_volta_da_estimativa_e_ordena_os_metodos():
    tabela = _avaliacoes()
    y = analise.rotulo(tabela)
    gerador = np.random.default_rng(1)
    escores = {"bom": tabela["insatisfacao"].to_numpy(), "ruido": gerador.normal(size=len(tabela))}
    resultado, comparacoes = analise.desempenho(y, escores, repeticoes=200)
    assert comparacoes.empty  # nenhum par pedido
    for coluna in analise.COLUNAS_DO_ALERTA:
        assert (resultado[f"{coluna} inf"] <= resultado[coluna]).all() and (resultado[coluna] <= resultado[f"{coluna} sup"]).all()
    assert resultado.loc["bom", "AUC"] > 0.85 and abs(resultado.loc["ruido", "AUC"] - 0.5) < 0.05
    assert resultado.loc["bom", "leitura para pegar 80%"] < resultado.loc["ruido", "leitura para pegar 80%"]
    assert resultado.loc["ruido", "leitura para pegar 80%"] == pytest.approx(0.8, abs=0.05)  # ao acaso, lê-se 80% para pegar 80%


def test_diferencas_declaram_empate_so_quando_o_intervalo_inclui_o_zero():
    tabela = _avaliacoes()
    y = analise.rotulo(tabela)
    gerador = np.random.default_rng(2)
    bom = tabela["insatisfacao"].to_numpy()
    escores = {"bom": bom, "quase igual": bom + gerador.normal(scale=0.02, size=len(bom)), "ruido": gerador.normal(size=len(bom))}
    por_metodo, resultado = analise.desempenho(y, escores, [("bom", "quase igual"), ("bom", "ruido")], repeticoes=200)
    assert bool(resultado.loc["bom − quase igual", "AUC empata"])
    assert not bool(resultado.loc["bom − ruido", "AUC empata"]) and resultado.loc["bom − ruido", "AUC inf"] > 0.3
    assert resultado.loc["bom − ruido", "AUC"] == pytest.approx(por_metodo.loc["bom", "AUC"] - por_metodo.loc["ruido", "AUC"])
    # pareado: o intervalo da diferença entre dois métodos quase iguais é bem mais estreito que o de cada um
    largura_da_diferenca = resultado.loc["bom − quase igual", "AUC sup"] - resultado.loc["bom − quase igual", "AUC inf"]
    assert largura_da_diferenca < 0.2 * (por_metodo.loc["bom", "AUC sup"] - por_metodo.loc["bom", "AUC inf"])


def test_metricas_do_alerta_batem_com_as_funcoes_de_metricas():
    tabela = _avaliacoes(n=800, semente=9)
    y, escore = analise.rotulo(tabela), np.round(tabela["insatisfacao"].to_numpy(), 1)  # com empates
    revocacao, precisao = metricas.captura_no_topo(y, escore, analise.PARCELA_LIDA)
    esperado = [metricas.auc(y, escore), precisao, revocacao, metricas.leitura_para_capturar(y, escore, analise.ALVO_DE_CAPTURA)]
    assert analise.metricas_do_alerta(y, escore) == pytest.approx(esperado)


def test_curvas_de_captura_tem_teto_acaso_e_faixa():
    tabela = _avaliacoes()
    y = analise.rotulo(tabela)
    curvas = analise.curvas_de_captura(y, {"modelo": tabela["insatisfacao"].to_numpy()}, com_faixa="modelo", repeticoes=100)
    assert len(curvas) == 41 and curvas["parcela lida"].iloc[[0, -1]].tolist() == [0.0, 1.0]
    assert (curvas["modelo"] <= curvas["teto"] + 1e-9).all() and (curvas["modelo"] >= curvas["acaso"] - 1e-9).all()
    assert (curvas["modelo inf"] <= curvas["modelo"] + 1e-9).all() and (curvas["modelo"] <= curvas["modelo sup"] + 1e-9).all()


def test_regra_da_nota_perfeita_quando_o_rotulo_e_a_propria_nota():
    tabela = _avaliacoes()
    nota = tabela["nota"].to_numpy()
    y = nota <= 2
    resultado = analise.regra_da_nota(y, nota, -nota.astype(float), repeticoes=50)
    regra = resultado.loc['Regra "nota até 2"']
    assert regra["precisão"] == 1.0 and regra["revocação"] == 1.0
    assert regra["parcela marcada"] == pytest.approx(y.mean())
    assert resultado.loc["Texto, marcando o mesmo volume", "precisão"] == pytest.approx(1.0)  # o escore é a nota: mesma pilha


def test_auc_por_rotulo_muda_com_o_gabarito_e_e_maior_onde_os_rotulos_concordam():
    tabela = _avaliacoes()
    escore = -tabela["nota"].to_numpy().astype(float) + np.random.default_rng(3).normal(scale=1e-3, size=len(tabela))
    resultado = analise.auc_por_rotulo(tabela, {"modelo": escore}, repeticoes=50).loc["modelo"]
    assert resultado.loc["nota até 2", "AUC"] == pytest.approx(1.0)  # o escore é a própria nota
    assert resultado.loc["não recomendaria", "AUC"] < 0.99
    assert resultado.loc["só onde os dois concordam", "AUC"] > resultado.loc["não recomendaria", "AUC"]
    assert resultado.loc["só onde os dois concordam", "avaliações"] < len(tabela)


def test_texto_acrescenta_a_nota_quando_carrega_informacao_propria():
    validacao, teste = _avaliacoes(semente=4), _avaliacoes(semente=5)
    escores = analise.nota_mais_texto(validacao, validacao["insatisfacao"].to_numpy(), teste, teste["insatisfacao"].to_numpy())
    y = analise.rotulo(teste)
    assert metricas.auc(y, escores["Nota + texto"]) > metricas.auc(y, escores["Só a nota"]) + 0.01
    por_nota = analise.texto_dentro_da_nota(teste, teste["insatisfacao"].to_numpy(), repeticoes=50)
    assert por_nota.index.tolist() == [1, 2, 3, 4, 5] and por_nota["avaliações"].sum() == len(teste)
    assert por_nota.loc[3, "AUC do texto"] > 0.6  # dentro da mesma nota, o escore ainda separa


def test_calibracao_corrige_modelo_confiante_demais():
    gerador = np.random.default_rng(6)
    logito = gerador.normal(-1.0, 2.0, size=(2, 20000))
    y = gerador.random((2, 20000)) < metricas.sigmoide(logito)
    resumo, curvas, recalibrada = analise.calibracao(y[0], 3.0 * logito[0], y[1], 3.0 * logito[1], repeticoes=50)
    assert resumo["platt_a"] == pytest.approx(1 / 3, abs=0.02)
    assert resumo["recalibrada"]["erro_de_calibracao"] < resumo["bruta"]["erro_de_calibracao"]
    assert resumo["diferenca_recalibrada_menos_bruta"]["superior"] < 0
    assert set(curvas["versão"]) == {"bruta", "recalibrada"} and len(curvas) == 20
    assert recalibrada.shape == (20000,)


def test_faixas_de_tamanho_e_de_idade():
    assert analise.faixa_de_tamanho(pd.Series([1, 5, 6, 20, 21, 41])).tolist() == ["até 5 palavras", "até 5 palavras", "6 a 10", "11 a 20", "21 a 40", "mais de 40"]
    idades = analise.faixa_de_idade(pd.Series([1995.0, 1980.0, 1950.0, np.nan, 59.0, 2015.0]))
    assert idades.tolist() == ["18 a 29", "30 a 39", "60 ou mais", "não informada", "não informada", "não informada"]


def test_erros_por_recorte_mostram_onde_o_escore_nao_funciona():
    tabela = _avaliacoes()
    y = analise.rotulo(tabela)
    escore = tabela["insatisfacao"].to_numpy().copy()
    de_entrega = tabela["texto"].str.contains("entrega").to_numpy()
    escore[de_entrega] = np.random.default_rng(7).normal(size=de_entrega.sum())  # nas avaliações de entrega, o escore é ruído
    resultado = analise.erros_por_recorte(tabela, y, escore, metricas.sigmoide(escore), repeticoes=50)
    assunto = resultado.loc["assunto"]
    assert abs(assunto.loc["fala de entrega ou prazo", "AUC"] - 0.5) < 0.06
    assert assunto.loc["não fala de entrega", "AUC"] > 0.85
    assert assunto["avaliações"].sum() == len(tabela)
    assert {"tamanho do texto", "gênero", "idade"} <= set(resultado.index.get_level_values("recorte"))


def test_leitura_manual_versionada_soma_45_de_cada_tipo_e_nao_tem_texto():
    resultado = analise.leitura_manual(dados.ARQUIVO_LEITURA)
    assert resultado.groupby(level="tipo")["avaliações"].sum().to_dict() == {"alarme falso": 45, "reclamação perdida": 45}
    assert resultado.groupby(level="tipo")["parcela"].sum().tolist() == pytest.approx([1.0, 1.0])
    bruto = pd.read_csv(dados.ARQUIVO_LEITURA)
    assert list(bruto.columns) == ["chave", "tipo", "categoria"] and bruto["chave"].str.fullmatch(r"[0-9a-f]{16}").all()


def test_leitura_no_modelo_final_refaz_a_selecao_pelos_escores():
    leitura = pd.read_csv(dados.ARQUIVO_LEITURA)
    falsos = leitura.loc[leitura["tipo"] == "alarme falso", "chave"].tolist()
    perdidas = leitura.loc[leitura["tipo"] == "reclamação perdida", "chave"].tolist()
    outros = [f"{i:016x}" for i in range(200)]
    teste = pd.DataFrame(
        {
            "chave": falsos + perdidas + outros,
            "nota": [5] * 45 + [1] * 45 + [5] * 100 + [1] * 100,
            "recomenda": [True] * 45 + [False] * 45 + [True] * 100 + [False] * 100,
        }
    )
    # escore que põe as lidas nos extremos: alarmes falsos no topo, reclamações perdidas no fundo
    escore = np.concatenate([np.full(45, 9.0), np.full(45, -9.0), np.linspace(-1, 1, 200)])
    resumo = analise.leitura_no_modelo_final(dados.ARQUIVO_LEITURA, teste, escore)
    assert resumo == {"alarme falso": {"lidas": 45, "ainda_entre_as_mais_extremas": 45}, "reclamação perdida": {"lidas": 45, "ainda_entre_as_mais_extremas": 45}}
    resumo = analise.leitura_no_modelo_final(dados.ARQUIVO_LEITURA, teste, -escore)  # escore invertido: nenhuma lida fica nos extremos
    assert resumo["alarme falso"]["ainda_entre_as_mais_extremas"] == 0 and resumo["reclamação perdida"]["ainda_entre_as_mais_extremas"] == 0


def test_auc_por_rotulo_bate_com_o_calculo_direto():
    tabela = _avaliacoes(n=1500, semente=8)
    escore = tabela["insatisfacao"].to_numpy()
    resultado = analise.auc_por_rotulo(tabela, {"modelo": escore}, repeticoes=30).loc["modelo"]
    nao, baixa = analise.rotulo(tabela, "recomendaria"), analise.rotulo(tabela, "nota")
    iguais = nao == baixa
    assert resultado.loc["não recomendaria", "AUC"] == pytest.approx(metricas.auc(nao, escore))
    assert resultado.loc["nota até 2", "AUC"] == pytest.approx(metricas.auc(baixa, escore))
    assert resultado.loc["só onde os dois concordam", "AUC"] == pytest.approx(metricas.auc(nao[iguais], escore[iguais]))
    assert resultado.loc["qualquer um dos dois sinais", "AUC"] == pytest.approx(metricas.auc(nao | baixa, escore))


def test_estabilidade_por_numero_de_temas_conta_os_pares():
    gerador = np.random.default_rng(0)
    matriz = sparse.csr_matrix(gerador.random((120, 30)) * (gerador.random((120, 30)) < 0.3))
    tabela = analise.estabilidade_por_numero_de_temas(matriz, numeros=(2, 3), repeticoes=3)
    assert tabela.index.tolist() == [2, 3] and tabela["pares de execuções"].tolist() == [3, 3]
    assert ((tabela["mínimo"] <= tabela["estabilidade"]) & (tabela["estabilidade"] <= tabela["máximo"])).all()
    assert analise.NUMERO_DE_TEMAS in analise.NUMEROS_DE_TEMAS


def test_motivos_nomeados_medidos_e_agrupados():
    # vocabulário com as âncoras de dois motivos; dois temas plantados
    nomes = np.array(["atraso", "defeito", "entrega", "prazo", "quebrado", "troca", "zzz"])
    pesos = np.array([[3.0, 0, 2.0, 1.0, 0, 0, 0], [0, 3.0, 0, 0, 2.0, 1.0, 0]])
    assert analise.motivo_de_cada_tema(pesos, nomes) == ["Atraso e prazo de entrega", "Defeito, avaria ou item errado"]
    documentos = sparse.csr_matrix(np.array([[1.0, 0, 1, 1, 0, 0, 0], [0, 1.0, 0, 0, 1, 1, 0], [1.0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]))
    indices = analise.motivo_de_cada_avaliacao(documentos, pesos, nomes)
    motivos = analise.lista_de_motivos()
    assert [motivos[i] for i in indices] == ["Atraso e prazo de entrega", "Defeito, avaria ou item errado", "Atraso e prazo de entrega", "Outros"]

    por_grupo = analise.motivos_por_grupo(indices, ["Móveis", "Móveis", "Livros", "Livros"], minimo=1)
    assert por_grupo.drop(columns="reclamações").sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert por_grupo.loc["Livros", "Outros"] == pytest.approx(0.5)

    familias = analise.parcela_por_familia(indices, repeticoes=50)
    assert familias["parcela"].to_dict() == pytest.approx({"entrega e loja": 0.5, "produto": 0.25, "sem motivo claro": 0.25})

    tamanhos = analise.tamanho_dos_motivos(pesos, [pesos, pesos[[1, 0]]], nomes, documentos, repeticoes=100)
    assert tamanhos["parcela"].sum() == pytest.approx(1.0)
    assert tamanhos.loc["Atraso e prazo de entrega", "parcela"] == pytest.approx(0.5)
    assert tamanhos.loc["Atraso e prazo de entrega", "execuções em que aparece"] == 2
    assert tamanhos.loc["Atraso e prazo de entrega", "estabilidade"] == pytest.approx(1.0)  # o mesmo tema nas duas execuções
    assert (tamanhos["amostragem inf"] <= tamanhos["parcela"]).all() and (tamanhos["parcela"] <= tamanhos["amostragem sup"]).all()


def test_todo_motivo_tem_familia_e_ancoras_normalizadas():
    assert set(analise.FAMILIAS) == set(analise.lista_de_motivos())
    for ancoras in analise.MOTIVOS.values():
        assert all(a == a.lower() and a.replace("_", "").isalpha() and a.isascii() for a in ancoras)
