import numpy as np
import pytest
from scipy import sparse

from nlp_avaliacoes import temas

NOMES = np.array(list("abcdefghij"))


def _pesos(*topos: str) -> np.ndarray:
    """Matriz tema x termo em que as letras de cada tema têm peso decrescente e as demais, zero."""
    matriz = np.zeros((len(topos), len(NOMES)))
    for linha, letras in enumerate(topos):
        for posicao, letra in enumerate(letras):
            matriz[linha, np.flatnonzero(NOMES == letra)[0]] = len(letras) - posicao
    return matriz


def test_jaccard_casos_conhecidos():
    assert temas.jaccard("abc", "bcd") == pytest.approx(0.5)  # 2 em comum, 4 na união
    assert temas.jaccard("abc", "abc") == 1.0
    assert temas.jaccard("abc", "def") == 0.0


def test_palavras_do_topo_seguem_o_peso():
    assert temas.palavras_do_topo(_pesos("cab", "jih"), NOMES, quantas=3) == [["c", "a", "b"], ["j", "i", "h"]]


def test_estabilidade_e_um_quando_os_temas_so_trocam_de_ordem():
    a = _pesos("abc", "def", "ghi")
    assert temas.estabilidade(a, a[[2, 0, 1]], quantas=3) == pytest.approx(1.0)


def test_estabilidade_e_zero_quando_nenhuma_palavra_se_repete():
    assert temas.estabilidade(_pesos("abc", "def"), _pesos("ghi", "jgh"), quantas=3) == pytest.approx(0.0)


def test_estabilidade_caso_feito_a_mao():
    # tema 1 igual (Jaccard 1); tema 2 troca uma de três palavras (2 em comum, 4 na união = 0,5): média 0,75
    a = _pesos("abc", "def")
    b = _pesos("deg", "abc")
    assert temas.estabilidade(a, b, quantas=3) == pytest.approx(0.75)
    par, valor = temas.emparelhar(temas.semelhanca_das_palavras(a, b, quantas=3))
    assert par.tolist() == [1, 0]
    assert valor == pytest.approx([1.0, 0.5])


def test_emparelhar_maximiza_o_total_e_nao_o_primeiro_par():
    # escolher primeiro o 0,9 obrigaria a fechar com 0,1 (total 1,0); o casamento ótimo soma 1,6
    semelhanca = np.array([[0.9, 0.8], [0.8, 0.1]])
    par, valor = temas.emparelhar(semelhanca)
    assert par.tolist() == [1, 0]
    assert valor.sum() == pytest.approx(1.6)


def test_estabilidade_media_e_por_tema():
    referencia = _pesos("abc", "def")
    execucoes = [_pesos("abc", "def"), _pesos("abc", "ghi"), _pesos("ghi", "abc")]
    por_tema = temas.estabilidade_por_tema(referencia, execucoes, quantas=3)
    assert por_tema == pytest.approx([1.0, 1 / 3])  # "abc" volta sempre; "def" só em uma de três
    media, minimo, maximo = temas.estabilidade_media(execucoes, quantas=3)
    assert (minimo, maximo) == (pytest.approx(0.5), pytest.approx(1.0))
    assert media == pytest.approx((0.5 + 0.5 + 1.0) / 3)


def test_semelhanca_de_cosseno_casos_conhecidos():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [0.0, 3.0]])
    esperado = np.array([[1.0, 0.0], [0.5**0.5, 0.5**0.5]])
    assert temas.semelhanca_de_cosseno(a, b) == pytest.approx(esperado)


def _corpus_com_temas_plantados(semente: int = 0, documentos: int = 600):
    """Três temas com vocabulários separados de 10 termos (mais 10 termos de ruído comuns a todos)."""
    gerador = np.random.default_rng(semente)
    tema = gerador.integers(0, 3, size=documentos)
    matriz = np.zeros((documentos, 40))
    for linha, t in enumerate(tema):
        proprios = gerador.choice(10, size=6, replace=False) + 10 * t
        matriz[linha, proprios] = gerador.uniform(0.5, 1.5, size=6)
        matriz[linha, 30 + gerador.choice(10, size=2, replace=False)] = 0.2
    matriz /= np.linalg.norm(matriz, axis=1, keepdims=True)
    return sparse.csr_matrix(matriz), tema


def test_nmf_recupera_temas_plantados_e_a_estabilidade_fica_alta():
    matriz, tema = _corpus_com_temas_plantados()
    execucoes = temas.execucoes_de_nmf(matriz, temas=3, repeticoes=4, semente=1)
    media, minimo, _ = temas.estabilidade_media(execucoes, quantas=10)
    assert minimo > 0.8 and media > 0.9
    # cada tema plantado tem um tema do NMF cujas 10 palavras do topo são o seu vocabulário
    plantado = np.zeros((3, 40))
    for t in range(3):
        plantado[t, 10 * t : 10 * t + 10] = 1.0
    assert temas.estabilidade(plantado, execucoes[0], quantas=10) > 0.9


def test_estabilidade_cai_quando_se_pede_mais_temas_do_que_existem():
    matriz, _ = _corpus_com_temas_plantados()
    certo = temas.estabilidade_media(temas.execucoes_de_nmf(matriz, 3, repeticoes=4, semente=1), quantas=10)[0]
    demais = temas.estabilidade_media(temas.execucoes_de_nmf(matriz, 8, repeticoes=4, semente=1), quantas=10)[0]
    assert demais < certo - 0.1


def test_kmeans_com_palavras_caracteristicas_recupera_os_grupos_plantados():
    matriz, tema = _corpus_com_temas_plantados(semente=2)
    pesos = temas.palavras_caracteristicas(matriz, tema, grupos=3)
    for t in range(3):
        assert set(np.argsort(-pesos[t])[:10]) == set(range(10 * t, 10 * t + 10))
    execucoes = temas.execucoes_de_kmeans(matriz.toarray(), matriz, grupos=3, repeticoes=3, semente=1)
    assert temas.estabilidade_media(execucoes, quantas=10)[0] > 0.9


def test_tema_dominante_e_parcelas():
    mistura = np.array([[0.1, 0.7], [0.0, 0.0], [0.4, 0.2], [0.9, 0.1]])
    dominante = temas.tema_dominante(mistura)
    assert dominante.tolist() == [1, -1, 0, 0]  # a linha zerada fica sem tema
    assert temas.parcelas(dominante, 2) == pytest.approx([0.5, 0.25])


def test_nomear_usa_as_ancoras_pesa_a_posicao_e_aceita_dois_temas_no_mesmo_motivo():
    topo = [
        ["entrega", "prazo", "recebi", "x"],
        ["defeito", "parou", "y", "z"],
        ["prazo", "entrega", "w", "k"],  # segundo tema do mesmo motivo
        ["q", "r", "s", "t"],  # nenhuma âncora
        ["defeito", "x", "y", "z"],  # uma âncora só: abaixo do mínimo
        ["quebrou", "parou", "entrega", "prazo"],  # 2 âncoras de cada: ganha o motivo com âncoras mais no topo
    ]
    ancoras = {"Entrega": ["entrega", "prazo", "recebi"], "Defeito": ["defeito", "parou", "quebrou"]}
    assert temas.nomear(topo, ancoras) == ["Entrega", "Defeito", "Entrega", "Outros", "Outros", "Defeito"]


def test_mistura_recupera_o_tema_de_cada_documento_sem_mexer_nos_temas():
    matriz, tema = _corpus_com_temas_plantados(semente=3)
    plantado = np.zeros((3, 40))
    for t in range(3):
        plantado[t, 10 * t : 10 * t + 10] = 1.0
    copia = plantado.copy()
    misturas = temas.mistura(matriz, plantado)
    assert np.array_equal(plantado, copia)
    assert (temas.tema_dominante(misturas) == tema).all()


def test_parcela_por_motivo_soma_os_temas_do_mesmo_motivo():
    dominante = np.array([0, 0, 1, 2, 2, 2, -1, 1])
    parcelas = temas.parcela_por_motivo(dominante, ["Entrega", "Defeito", "Entrega"], ["Entrega", "Defeito", "Outros"])
    assert parcelas == pytest.approx([5 / 8, 2 / 8, 0.0])
