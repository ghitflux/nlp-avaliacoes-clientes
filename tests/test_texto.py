import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nlp_avaliacoes import texto

CORPUS = [
    "produto otimo chegou antes do prazo",
    "nao recebi o produto ate agora",
    "produto veio com defeito e nao funciona",
    "otimo otimo otimo recomendo",
    "pessimo nao recomendo nao comprem",
    "chegou no prazo produto bom",
]


def _alinhar(nossa, vetorizador_sklearn, nomes) -> np.ndarray:
    """Matriz do scikit-learn com as colunas na ordem dos nossos termos (o par de palavras aqui usa "_")."""
    coluna = vetorizador_sklearn.vocabulary_
    assert len(coluna) == nossa.shape[1]
    return np.array([coluna[nome.replace("_", " ")] for nome in nomes])


def test_normalizar_tira_acento_pontuacao_e_letra_repetida():
    entrada = pd.Series(["Péssimooo!!!  Não   recebi.", "ÓTIMO, chegou em 3 dias", "!!!"])
    assert texto.normalizar(entrada).tolist() == ["pessimo nao recebi", "otimo chegou em dias", ""]


def test_termos_formam_pares_depois_de_tirar_as_palavras_ignoradas():
    assert texto.termos("nao recebi o produto") == ["nao", "recebi", "o", "produto", "nao_recebi", "recebi_o", "o_produto"]
    assert texto.termos("nao recebi o produto", ignorar=frozenset({"o"})) == ["nao", "recebi", "produto", "nao_recebi", "recebi_produto"]
    assert texto.termos("nao recebi", pares=False) == ["nao", "recebi"]


def test_palavras_vazias_nao_incluem_negacao():
    assert not {"nao", "nunca", "nem", "sem", "mas"} & texto.PALAVRAS_VAZIAS


def test_idf_suavizado_caso_feito_a_mao():
    # 3 documentos; "a" está nos 3, "b" em 1: idf = ln((1+3)/(1+df)) + 1
    listas = [["a", "b"], ["a"], ["a"]]
    vocabulario = texto.montar_vocabulario(listas)
    idf = texto.idf_suavizado(texto.contar(listas, vocabulario))
    assert idf[vocabulario["a"]] == pytest.approx(1.0)
    assert idf[vocabulario["b"]] == pytest.approx(np.log(2.0) + 1.0)


def test_ponderar_caso_feito_a_mao():
    # documento "a a b" com idf(a) = 1 e idf(b) = 2: (2*1, 1*2) = (2, 2), que normalizado dá (0,7071; 0,7071)
    listas = [["a", "a", "b"]]
    contagens = texto.contar(listas, {"a": 0, "b": 1})
    assert contagens.toarray().tolist() == [[2.0, 1.0]]
    pesos = texto.ponderar(contagens, np.array([1.0, 2.0])).toarray()
    assert pesos == pytest.approx(np.array([[0.5**0.5, 0.5**0.5]]))
    # sublinear: tf vira 1 + ln(2) para "a"
    sub = texto.ponderar(contagens, np.array([1.0, 1.0]), sublinear=True).toarray()[0]
    assert sub[0] / sub[1] == pytest.approx(1.0 + np.log(2.0))


@pytest.mark.parametrize("sublinear", [False, True])
@pytest.mark.parametrize("df_minimo", [1, 2])
def test_tfidf_de_palavras_bate_com_o_scikit_learn(sublinear, df_minimo):
    nosso = texto.TfidfDePalavras(df_minimo=df_minimo, sublinear=sublinear)
    matriz = nosso.ajustar(CORPUS)
    referencia = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", min_df=df_minimo, sublinear_tf=sublinear)
    esperado = referencia.fit_transform(CORPUS).toarray()
    colunas = _alinhar(matriz, referencia, nosso.nomes)
    assert matriz.toarray() == pytest.approx(esperado[:, colunas], abs=1e-12)
    # texto novo, com palavra nunca vista: mesma resposta do scikit-learn e nenhuma coluna nova
    novos = ["produto pessimo chegou quebrado", "xyz"]
    assert nosso.transformar(novos).toarray() == pytest.approx(referencia.transform(novos).toarray()[:, colunas], abs=1e-12)


def test_linhas_do_tfidf_tem_norma_um_e_documento_sem_termo_conhecido_fica_zerado():
    nosso = texto.TfidfDePalavras(df_minimo=1)
    matriz = nosso.ajustar(CORPUS)
    assert np.sqrt(matriz.multiply(matriz).sum(axis=1)) == pytest.approx(np.ones((len(CORPUS), 1)))
    assert nosso.transformar(["zzz qqq"]).nnz == 0


def test_vocabulario_e_idf_so_veem_o_treino():
    nosso = texto.TfidfDePalavras(df_minimo=1)
    nosso.ajustar(CORPUS[:3])
    vocabulario, idf = dict(nosso.vocabulario), nosso.idf.copy()
    nosso.transformar(CORPUS[3:])  # contém "pessimo" e "recomendo", que o treino não viu
    assert nosso.vocabulario == vocabulario and "pessimo" not in nosso.vocabulario
    assert np.array_equal(nosso.idf, idf)
    with pytest.raises(RuntimeError):
        texto.TfidfDePalavras().transformar(CORPUS)


def test_vocabulario_respeita_df_minimo_e_maximo():
    listas = [["a", "b", "c"], ["a", "b"], ["a"]]
    assert list(texto.montar_vocabulario(listas, df_minimo=2)) == ["a", "b"]
    assert list(texto.montar_vocabulario(listas, maximo=1)) == ["a"]


def test_ngramas_de_caracteres_caso_feito_a_mao():
    # " ab " tem dois trigramas (" ab", "ab ") e um 4-grama (" ab "); nada de 5 letras
    documento, codigo = texto.codigos_de_caracteres(["ab"], 3, 5)
    assert documento.tolist() == [0, 0, 0]
    assert sorted(texto.decodificar(c) for c in codigo) == [" ab", " ab ", "ab "]


def test_ngramas_de_caracteres_nao_cruzam_documentos():
    documento, codigo = texto.codigos_de_caracteres(["ab", "cd"], 3, 5)
    gramas = {(int(d), texto.decodificar(c)) for d, c in zip(documento, codigo)}
    assert gramas == {(0, " ab"), (0, "ab "), (0, " ab "), (1, " cd"), (1, "cd "), (1, " cd ")}


def test_contagem_de_caracteres_bate_com_o_scikit_learn():
    matriz, vocabulario = texto.contar_caracteres(CORPUS, lote=4)  # lote pequeno: exercita a junção dos lotes
    referencia = CountVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False)
    esperado = referencia.fit_transform([f" {t} " for t in CORPUS])
    colunas = np.array([referencia.vocabulary_[texto.decodificar(c)] for c in vocabulario])
    assert matriz.shape == esperado.shape
    assert np.array_equal(matriz.toarray(), esperado.toarray()[:, colunas])


def test_tfidf_de_caracteres_bate_com_o_scikit_learn_e_ignora_ngrama_novo():
    nosso = texto.TfidfDeCaracteres(df_minimo=2)
    matriz = nosso.ajustar(CORPUS)
    referencia = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False, min_df=2, sublinear_tf=True)
    esperado = referencia.fit_transform([f" {t} " for t in CORPUS]).toarray()
    colunas = np.array([referencia.vocabulary_[texto.decodificar(c)] for c in nosso.vocabulario])
    assert matriz.toarray() == pytest.approx(esperado[:, colunas], abs=1e-12)
    novos = ["produto pessimo chegou quebrado", "xyzw"]
    esperado_novos = referencia.transform([f" {t} " for t in novos]).toarray()[:, colunas]
    assert nosso.transformar(novos).toarray() == pytest.approx(esperado_novos, abs=1e-12)


def test_codigos_de_caracteres_recusam_texto_nao_normalizado():
    with pytest.raises(ValueError):
        texto.codigos_de_caracteres(["Não"])


def test_escore_do_lexico_conta_palavras_e_pares():
    escore = texto.escore_do_lexico(["produto otimo", "pessimo nao recebi", "nao recomendo pessimo lixo"])
    assert escore.tolist() == [0.0, 2.0, 3.0]
