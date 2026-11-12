import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from nlp_avaliacoes import metricas


def test_auc_casos_conhecidos():
    y = np.array([0, 0, 1, 1])
    assert metricas.auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)  # exemplo clássico
    assert metricas.auc(y, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)
    assert metricas.auc(y, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(0.0)
    assert metricas.auc(y, np.zeros(4)) == pytest.approx(0.5)  # tudo empatado: acaso
    assert np.isnan(metricas.auc(np.ones(4), np.arange(4.0)))  # sem negativos não há AUC


def test_auc_bate_com_o_scikit_learn_inclusive_com_empates():
    gerador = np.random.default_rng(0)
    y = gerador.random(500) < 0.3
    escore = np.round(gerador.normal(size=500) + y, 1)  # arredondado: muitos empates
    assert metricas.auc(y, escore) == pytest.approx(roc_auc_score(y, escore), abs=1e-12)


@pytest.mark.parametrize("casas", [1, 6])  # com muitos empates e praticamente sem empates
def test_auc_e_capturas_com_uma_ordenacao_so_bate_com_as_funcoes_separadas(casas):
    gerador = np.random.default_rng(11)
    y = gerador.random(700) < 0.3
    escore = np.round(gerador.normal(size=700) + 1.2 * y, casas)
    fracoes = np.array([0.0, 0.05, 0.1, 0.33, 0.8, 1.0])
    area, revocacao, precisao = metricas.auc_e_capturas(y, escore, fracoes)
    esperada_revocacao, esperada_precisao = metricas.capturas(y, escore, fracoes)
    assert area == pytest.approx(roc_auc_score(y, escore), abs=1e-12)
    assert revocacao == pytest.approx(esperada_revocacao, abs=1e-12)
    assert precisao[1:] == pytest.approx(esperada_precisao[1:], abs=1e-12)  # a precisão de ler 0% é indefinida nas duas
    assert np.isnan(metricas.auc_e_capturas(np.ones(5), np.arange(5.0), fracoes)[0])


def test_captura_no_topo_sem_empate():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    escore = np.arange(10, 0, -1.0)
    revocacao, precisao = metricas.captura_no_topo(y, escore, 0.2)  # lê as 2 primeiras, as 2 são reclamação
    assert (revocacao, precisao) == (pytest.approx(2 / 3), pytest.approx(1.0))
    revocacao, precisao = metricas.captura_no_topo(y, escore, 0.5)  # lê 5, pega as 3
    assert (revocacao, precisao) == (pytest.approx(1.0), pytest.approx(0.6))


def test_captura_no_topo_com_empate_usa_o_valor_esperado():
    # 4 avaliações empatadas no topo (3 reclamações) e só 2 vagas: espera-se pegar 2 * 3/4 = 1,5
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    escore = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0.0])
    revocacao, precisao = metricas.captura_no_topo(y, escore, 0.2)
    assert (revocacao, precisao) == (pytest.approx(1.5 / 4), pytest.approx(0.75))
    # escore constante equivale a ler ao acaso: revocação = parcela lida, precisão = prevalência
    revocacao, precisao = metricas.captura_no_topo(y, np.zeros(10), 0.3)
    assert (revocacao, precisao) == (pytest.approx(0.3), pytest.approx(0.4))


def test_curva_de_captura_vai_de_zero_a_um_e_nunca_desce():
    gerador = np.random.default_rng(1)
    y = gerador.random(300) < 0.25
    escore = gerador.normal(size=300) + 1.5 * y
    fracoes = np.linspace(0, 1, 41)
    curva = metricas.curva_de_captura(y, escore, fracoes)
    assert curva[0] == pytest.approx(0.0) and curva[-1] == pytest.approx(1.0)
    assert (np.diff(curva) >= -1e-12).all()
    assert (curva >= fracoes - 1e-12).all()  # melhor que o acaso em toda a curva, neste exemplo


def test_leitura_para_capturar_com_escore_perfeito_e_a_prevalencia():
    y = np.array([1] * 30 + [0] * 70)
    escore = np.arange(100, 0, -1.0)
    assert metricas.leitura_para_capturar(y, escore, 1.0) == pytest.approx(0.30)
    assert metricas.leitura_para_capturar(y, escore, 0.5) == pytest.approx(0.15)
    assert metricas.leitura_para_capturar(y, np.zeros(100), 0.8) == pytest.approx(0.8)  # ao acaso


def test_precisao_e_revocacao_caso_feito_a_mao():
    y = np.array([1, 1, 1, 0, 0])
    previsto = np.array([1, 1, 0, 1, 0])
    assert metricas.precisao_e_revocacao(y, previsto) == (pytest.approx(2 / 3), pytest.approx(2 / 3))


def test_bootstrap_da_media_bate_com_a_teoria_normal():
    gerador = np.random.default_rng(2)
    amostra = gerador.normal(10.0, 2.0, size=400)
    inferior, superior = metricas.bootstrap(np.mean, amostra, repeticoes=4000, semente=3)
    erro_padrao = amostra.std(ddof=1) / np.sqrt(400)
    assert inferior == pytest.approx(amostra.mean() - 1.96 * erro_padrao, abs=0.02)
    assert superior == pytest.approx(amostra.mean() + 1.96 * erro_padrao, abs=0.02)


def test_bootstrap_cobre_o_valor_verdadeiro_na_frequencia_prometida():
    gerador = np.random.default_rng(4)
    cobriu = 0
    for i in range(300):
        amostra = gerador.normal(0.0, 1.0, size=80)
        inferior, superior = metricas.bootstrap(np.mean, amostra, repeticoes=300, semente=i)
        cobriu += inferior <= 0.0 <= superior
    assert 0.90 <= cobriu / 300 <= 0.985  # prometido: 95%


def test_bootstrap_e_reprodutivel_e_pareado():
    gerador = np.random.default_rng(5)
    y = gerador.random(400) < 0.3
    a = gerador.normal(size=400) + 2 * y
    b = a + gerador.normal(scale=0.01, size=400)  # quase o mesmo método

    def diferenca(y_, a_, b_):
        return metricas.auc(y_, a_) - metricas.auc(y_, b_)

    primeiro = metricas.bootstrap(diferenca, y, a, b, repeticoes=200, semente=7)
    assert primeiro == metricas.bootstrap(diferenca, y, a, b, repeticoes=200, semente=7)
    assert primeiro[0] < 0 < primeiro[1] and primeiro[1] - primeiro[0] < 0.01  # pareado: a diferença é estreita
    assert metricas.bootstrap(diferenca, y, a, a, repeticoes=50) == (0.0, 0.0)  # método contra ele mesmo
    solto = metricas.bootstrap(metricas.auc, y, a, repeticoes=200, semente=7)
    assert solto[0] < metricas.auc(y, a) < solto[1]


def test_intervalo_de_wilson_valor_de_referencia():
    inferior, superior = metricas.intervalo_de_wilson(10, 100)
    assert (inferior, superior) == (pytest.approx(0.0552, abs=5e-4), pytest.approx(0.1744, abs=5e-4))
    inferior, superior = metricas.intervalo_de_wilson(0, 20)
    assert inferior == 0.0 and 0.10 < superior < 0.20  # não colapsa em zero
    assert all(np.isnan(v) for v in metricas.intervalo_de_wilson(0, 0))


def test_brier_e_erro_de_calibracao_casos_conhecidos():
    assert metricas.brier(np.array([1, 0]), np.array([0.8, 0.3])) == pytest.approx(0.065)
    gerador = np.random.default_rng(6)
    probabilidade = gerador.random(40000)
    y = gerador.random(40000) < probabilidade  # calibrado por construção
    assert metricas.erro_de_calibracao(y, probabilidade) < 0.01
    exagerada = np.clip(probabilidade * 1.5, 0, 1)  # promete mais do que entrega
    assert metricas.erro_de_calibracao(y, exagerada) > 0.10
    prevista, observada, tamanho = metricas.curva_de_calibracao(y, probabilidade, faixas=10)
    assert tamanho.sum() == 40000 and (np.diff(prevista) > 0).all()
    assert observada == pytest.approx(prevista, abs=0.03)


def test_platt_recupera_a_escala_verdadeira():
    gerador = np.random.default_rng(8)
    logito_verdadeiro = gerador.normal(-1.0, 2.0, size=60000)
    y = gerador.random(60000) < metricas.sigmoide(logito_verdadeiro)
    confiante_demais = 2.0 * logito_verdadeiro + 1.0  # o modelo exagera e desloca
    a, b = metricas.ajustar_platt(confiante_demais, y)
    assert a == pytest.approx(0.5, abs=0.02)
    assert b == pytest.approx(-0.5, abs=0.04)
    calibrada = metricas.sigmoide(a * confiante_demais + b)
    assert metricas.erro_de_calibracao(y, calibrada) < metricas.erro_de_calibracao(y, metricas.sigmoide(confiante_demais))
