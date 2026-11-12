import numpy as np
import pandas as pd

from nlp_avaliacoes import dados


def _bruto() -> pd.DataFrame:
    """Oito linhas que exercitam cada descarte da limpeza."""
    linhas = [
        # data, avaliador, produto, título, nota, recomenda, texto
        ("2018-01-05 10:00:00", "ana", "p1", "Ótimo", 5, "Yes", "Chegou antes do prazo!"),
        ("2018-01-06 10:00:00", "ana", "p1", "Ótimo", 5, "Yes", "Chegou antes do prazo!"),  # mesma avaliação repetida
        ("2018-02-01 09:00:00", "bia", "p2", "Ruim", 1, "No", "Não recebi o produto."),
        ("2018-04-10 09:00:00", "caio", "p3", "Ruim", 1, "No", "NÃO RECEBI O PRODUTO"),  # mesmo texto, outro avaliador
        ("2018-04-11 09:00:00", "davi", "p4", "Sem texto", 3, "Yes", None),  # vazio
        ("2018-04-12 09:00:00", "eva", "p5", "Pontuação", 3, "Yes", "!!! 123"),  # vazio depois de normalizar
        ("2018-05-02 09:00:00", "fabio", "p6", "Bom", 4, None, "Gostei bastante."),  # sem resposta em recomendaria
        ("2018-05-20 09:00:00", "gil", "p7", "Regular", 2, "Yes", "Veio com defeito, mas trocaram."),
    ]
    tabela = pd.DataFrame(linhas, columns=["submission_date", "reviewer_id", "product_id", "review_title", "overall_rating", "recommend_to_a_friend", "review_text"])
    tabela["site_category_lv1"] = "Móveis"
    tabela["reviewer_birth_year"] = 1980.0
    tabela["reviewer_gender"] = ["F", "F", "F", "M", "M", "F", "M", None]
    tabela["reviewer_state"] = ["SP", "SP", "BA", "RS", "SP", "SP", "SP", None]
    return tabela.sample(frac=1.0, random_state=0)  # fora de ordem, de propósito


def test_limpar_conta_cada_descarte_e_mantem_a_primeira_ocorrencia():
    limpa, contagem = dados.limpar(_bruto())
    assert contagem == {
        "linhas_no_arquivo": 8,
        "sem_resposta_em_recomendaria": 1,
        "texto_vazio": 2,
        "mesma_avaliacao_repetida": 1,
        "mesmo_texto_em_outra_avaliacao": 1,
        "avaliacoes_usadas": 3,
    }
    assert limpa["data"].is_monotonic_increasing
    assert limpa["texto"].tolist() == ["otimo chegou antes do prazo", "ruim nao recebi o produto", "regular veio com defeito mas trocaram"]
    assert limpa["data"].dt.strftime("%m-%d").tolist() == ["01-05", "02-01", "05-20"]  # ficou a primeira no tempo
    assert limpa["recomenda"].tolist() == [True, False, True]
    assert limpa["palavras"].tolist() == [4, 4, 5]  # palavras do corpo, sem o título


def test_periodos_seguem_a_data_e_nenhum_texto_aparece_em_dois_periodos():
    limpa, _ = dados.limpar(_bruto())
    assert limpa["periodo"].tolist() == ["treino", "treino", "teste"]
    datas = pd.Series(pd.to_datetime(["2018-03-31 23:59:59", "2018-04-01 00:00:00", "2018-04-30 23:59:59", "2018-05-01 00:00:00"]))
    assert dados.periodo(datas).tolist() == ["treino", "validação", "validação", "teste"]
    assert not limpa["texto"].duplicated().any()


def test_colunas_demograficas_viram_grupos_agregados_e_a_chave_nao_expoe_o_avaliador():
    limpa, _ = dados.limpar(_bruto())
    assert limpa["regiao"].tolist() == ["Sudeste", "Nordeste", "não informada"]
    assert limpa["genero"].tolist() == ["F", "F", "não informado"]
    assert limpa["chave"].str.fullmatch(r"[0-9a-f]{16}").all() and limpa["chave"].is_unique
    assert "reviewer_id" not in limpa.columns and limpa["avaliador"].dtype == np.int32
    de_novo, _ = dados.limpar(_bruto())
    assert de_novo["chave"].tolist() == limpa["chave"].tolist()  # estável entre execuções


def _tabela_grande(n: int = 40000) -> pd.DataFrame:
    gerador = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "chave": [f"{i:016x}" for i in range(n)],
            "periodo": np.where(np.arange(n) < 0.6 * n, "treino", np.where(np.arange(n) < 0.8 * n, "validação", "teste")),
            "nota": gerador.integers(1, 6, size=n),
            "recomenda": gerador.random(n) < 0.7,
        }
    )


def test_amostra_para_densa_e_fixa_sem_repeticao_e_respeita_os_periodos():
    tabela = _tabela_grande()
    amostra = dados.amostra_para_densa(tabela)
    assert (amostra["parte"] == "treino").sum() == 9000 and (amostra["parte"] == "teste").sum() == 5000
    assert amostra.index.is_unique
    assert (amostra.loc[amostra["parte"] == "teste", "periodo"] == "teste").all()
    assert (amostra.loc[amostra["parte"] != "teste", "periodo"] == "treino").all()
    a_mais = amostra[amostra["parte"] == "reclamações a mais"]
    assert len(a_mais) > 0 and (a_mais["nota"] <= 2).all() and not a_mais["recomenda"].any()
    assert dados.amostra_para_densa(tabela).index.equals(amostra.index)  # mesma semente, mesma amostra


def test_carregar_embeddings_sem_arquivo_devolve_none():
    assert dados.carregar_embeddings(pd.Series(["a"]), dados.BRUTOS / "arquivo_que_nao_existe.npz") is None
