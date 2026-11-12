"""A análise inteira num corpus sintético pequeno, em memória: confere o encadeamento, sem depender da base real."""

import json

import numpy as np
import pandas as pd
import pytest

from nlp_avaliacoes import analise, dados, pipeline, resumo_site

FRASES_DE_RECLAMACAO = [
    "nao recebi o produto aguardando ate hoje nao chegou",
    "entrega atrasou prazo passou dias de atraso demora",
    "veio com defeito quebrado quero troca devolucao",
    "aparelho parou de funcionar bateria fraca nao funciona no uso",
    "qualidade pessima material fragil acabamento ruim",
    "diferente da foto tamanho pequeno cor errada propaganda enganosa",
    "loja nao responde contato atendimento americanas site",
    "pessimo horrivel dinheiro jogado fora nao recomendo lixo",
]
FRASES_DE_ELOGIO = [
    "otimo produto chegou rapido recomendo",
    "excelente qualidade adorei superou expectativas",
    "muito bom entrega antes do prazo perfeito",
    "gostei bastante funciona bem vale a pena",
]
ENCHIMENTO = [f"pal{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(200)]


def _letras(numero: int) -> str:
    """Código só de letras, único por avaliação (a normalização descarta dígitos)."""
    letras = []
    for _ in range(4):
        numero, resto = divmod(numero, 25)
        letras.append(chr(97 + resto))  # de "a" a "y"
    return "x" + "z".join(letras)  # o "z" entre as letras impede três iguais seguidas, que a normalização juntaria


@pytest.fixture(scope="module")
def resultados() -> pipeline.Resultados:
    gerador = np.random.default_rng(0)
    n = 6000
    reclamacao = gerador.random(n) < 0.3
    linhas = []
    datas = pd.Timestamp("2018-01-01") + pd.to_timedelta(np.sort(gerador.integers(0, 150 * 24 * 3600, size=n)), unit="s")
    for i in range(n):
        frases = FRASES_DE_RECLAMACAO if reclamacao[i] else FRASES_DE_ELOGIO
        frase = frases[gerador.integers(len(frases))]
        if gerador.random() < 0.15:  # texto misto: um pouco do outro lado
            outras = FRASES_DE_ELOGIO if reclamacao[i] else FRASES_DE_RECLAMACAO
            frase += " mas " + outras[gerador.integers(len(outras))]
        enchimento = " ".join(gerador.choice(ENCHIMENTO, size=3))
        nota = int(gerador.choice([1, 2, 3], p=[0.6, 0.3, 0.1])) if reclamacao[i] else int(gerador.choice([3, 4, 5], p=[0.1, 0.4, 0.5]))
        recomenda = (not reclamacao[i]) if gerador.random() > 0.05 else bool(reclamacao[i])  # 5% de rótulo trocado
        linhas.append(
            {
                "submission_date": str(datas[i]), "reviewer_id": f"r{gerador.integers(4000)}", "product_id": f"p{i}",
                "site_category_lv1": str(gerador.choice(["Móveis", "Celulares", "Livros"])), "review_title": "titulo",
                "overall_rating": nota, "recommend_to_a_friend": "Yes" if recomenda else "No",
                "review_text": f"{frase} {enchimento} {_letras(i)}", "reviewer_birth_year": float(gerador.integers(1950, 2000)),
                "reviewer_gender": str(gerador.choice(["F", "M"])), "reviewer_state": str(gerador.choice(["SP", "BA", "RS"])),
            }
        )
    tabela, contagem = dados.limpar(pd.DataFrame(linhas))
    assert contagem["avaliacoes_usadas"] == n
    return pipeline.executar(tabela, contagem, repeticoes=30, falar=False)


def test_o_modelo_aprende_e_supera_o_acaso_no_corpus_sintetico(resultados):
    r = resultados
    assert r.desempenho.loc[pipeline.PALAVRAS, "AUC"] > 0.9
    assert r.desempenho.loc[pipeline.CARACTERES, "AUC"] > 0.9
    assert r.desempenho.loc[pipeline.LEXICO, "AUC"] > 0.6
    assert set(r.diferencas.index) == {f"{pipeline.PALAVRAS} − {pipeline.LEXICO}", f"{pipeline.PALAVRAS} − {pipeline.CARACTERES}"}
    assert r.base["periodos"]["treino"]["ate"] < "2018-04-01" <= r.base["periodos"]["validação"]["de"]
    assert r.base["periodos"]["validação"]["ate"] < "2018-05-01" <= r.base["periodos"]["teste"]["de"]


def test_os_motivos_plantados_sao_encontrados_e_as_parcelas_fecham(resultados):
    r = resultados
    assert r.motivos["parcela"].sum() == pytest.approx(1.0)
    assert r.familias["parcela"].sum() == pytest.approx(1.0)
    assert (r.motivos["temas"] > 0).sum() >= 6  # oito motivos plantados; a maioria tem de aparecer com nome
    assert r.por_categoria.drop(columns="reclamações").sum(axis=1).tolist() == pytest.approx([1.0] * len(r.por_categoria))
    assert r.estabilidade.index.tolist() == list(analise.NUMEROS_DE_TEMAS)
    assert r.estabilidade.loc[r.numero_de_temas, "pares de execuções"] == 45
    assert 0.0 <= r.estabilidade_do_modelo["estabilidade"] <= 1.0


def test_sem_o_arquivo_de_embeddings_a_comparacao_roda_so_com_o_lsa(resultados):
    r = resultados  # as chaves sintéticas não existem em nenhum arquivo de embeddings: a via densa cai no LSA
    assert not r.densa_disponivel
    assert list(r.representacoes.index) == [pipeline.AMOSTRA_TFIDF, pipeline.AMOSTRA_LSA, pipeline.AMOSTRA_BASE_INTEIRA]
    assert list(r.agrupamentos.index) == ["NMF sobre TF-IDF", pipeline.KMEANS_LSA]


def test_bloco_do_site_segue_as_regras_e_cabe_em_json(resultados):
    bloco = resumo_site.para_o_site(resultados)
    resumo_site.conferir(bloco)
    assert bloco["grafico"]["tipo"] == "linhas" and len(bloco["grafico"]["series"]) == 3
    assert bloco["grafico"]["series"][0]["papel"] == "principal" and bloco["grafico"]["series"][1]["papel"] == "base"
    assert len(bloco["grafico"]["faixa"]["inferior"]) == len(bloco["grafico"]["series"][0]["pontos"]) == 41
    assert "," in bloco["numero_chave"]["pt"] and "." in bloco["numero_chave"]["en"]  # decimal de cada idioma
    json.dumps({"para_o_site": bloco, "achados": resumo_site.achados(resultados)}, default=float)


def test_graficos_extras_seguem_a_especificacao_do_site(resultados):
    extras = resumo_site.para_o_site(resultados)["graficos_extras"]
    assert 3 <= len(extras) <= 5
    for grafico in extras:
        resumo_site.conferir_grafico(grafico, com_leitura=True)
        assert set(grafico["leitura"]) == {"pt", "en", "es"} and all(len(t) <= 90 for t in grafico["titulo"].values())
    assert [g["tipo"] for g in extras] == ["barras", "barras", "linhas", "linhas", "barras"]
    motivos = extras[1]["series"][1]["pontos"]
    assert sum(valor for _, valor in motivos) == pytest.approx(1.0, abs=0.01)  # as parcelas dos motivos fecham 100%
    with pytest.raises(AssertionError):  # a conferência recusa um gráfico sem a leitura
        resumo_site.conferir_grafico({k: v for k, v in extras[0].items() if k != "leitura"}, com_leitura=True)


def test_formatos_por_idioma():
    assert [resumo_site._pct(0.256, i) for i in resumo_site.IDIOMAS] == ["26%", "26%", "26 %"]
    assert [resumo_site._pct(0.256, i, 1) for i in resumo_site.IDIOMAS] == ["25,6%", "25.6%", "25,6 %"]
    assert [resumo_site._num(0.963, i) for i in resumo_site.IDIOMAS] == ["0,96", "0.96", "0,96"]
    assert [resumo_site._milhares(126152, i) for i in resumo_site.IDIOMAS] == ["126 mil", "126,000", "126 000"]
    assert set(resumo_site.MOTIVOS_TRADUZIDOS) == set(analise.lista_de_motivos())
