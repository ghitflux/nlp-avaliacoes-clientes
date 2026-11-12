"""Figuras do estudo. Azul é sempre o modelo de texto (método principal); laranja, a linha de base
(léxico ou leitura pela nota); verde-água, a alternativa (representação densa)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from nlp_avaliacoes.estilo import (
    APAGADO, AQUA, AZUL, EIXO, ESTILO, LARANJA, RAMPA_AZUL, SERIES, SUPERFICIE, TINTA, TINTA_SECUNDARIA, nova_figura, numero, percentual,
)

AMARELO = SERIES[3]
_PCT = FuncFormatter(lambda v, _: percentual(v))


def curva_de_captura(curvas: pd.DataFrame, modelo: str, lexico: str, parcela_lida: float) -> Figure:
    """Parcela das reclamações capturadas conforme a parcela das avaliações lidas."""
    with plt.rc_context(ESTILO):
        ponto = curvas.loc[np.isclose(curvas["parcela lida"], parcela_lida)].iloc[0]
        alvo = 0.80
        ler_modelo = float(np.interp(alvo, curvas[modelo], curvas["parcela lida"]))
        ler_lexico = float(np.interp(alvo, curvas[lexico], curvas["parcela lida"]))
        fig, ax = nova_figura(
            f"Para pegar 80% das reclamações: ler {percentual(ler_modelo)} com o modelo ou {percentual(ler_lexico)} com o léxico",
            "Parcela das reclamações que cai na pilha lida, quando o time lê primeiro as avaliações de maior escore (teste: maio de 2018).\n"
            f"Lendo {percentual(parcela_lida)}, o modelo pega {percentual(ponto[modelo])} das reclamações e o léxico, {percentual(ponto[lexico])} "
            f"(o máximo possível é {percentual(ponto['teto'])}).",
            altura=4.9,
        )
        fig.subplots_adjust(bottom=0.13, top=0.79)
        x = curvas["parcela lida"]
        ax.plot(x, curvas["teto"], color=EIXO, linewidth=1.2, zorder=1, label="teto: ordenação perfeita")
        ax.fill_between(x, curvas[f"{modelo} inf"], curvas[f"{modelo} sup"], color=AZUL, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, curvas[modelo], color=AZUL, zorder=4, label="modelo de texto (TF-IDF + regressão logística), com faixa de 95%")
        ax.plot(x, curvas[lexico], color=LARANJA, zorder=3, label="léxico de palavras negativas (sem treino)")
        ax.plot(x, curvas["acaso"], color=APAGADO, linewidth=1.2, zorder=1, label="ler ao acaso")
        ax.plot([ler_modelo, ler_lexico], [alvo, alvo], color=EIXO, linewidth=1.0, zorder=1)
        ax.plot(ler_modelo, alvo, "o", color=AZUL, zorder=5)
        ax.plot(ler_lexico, alvo, "o", color=LARANJA, zorder=5)
        ax.annotate(percentual(ler_modelo), (ler_modelo, alvo), xytext=(-9, 5), textcoords="offset points", ha="right", color=TINTA, fontsize=9, fontweight="bold")
        ax.annotate(percentual(ler_lexico), (ler_lexico, alvo), xytext=(9, -13), textcoords="offset points", ha="left", color=TINTA, fontsize=9, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.0, 0.02))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(_PCT)
        ax.yaxis.set_major_formatter(_PCT)
        ax.set_xlabel("parcela das avaliações lidas")
        ax.set_ylabel("parcela das reclamações capturadas")
        return fig


def _pontos_com_intervalo(ax, linhas: list[dict], casas: int = 3) -> None:
    """Ponto e intervalo de 95% por linha; cada item traz rotulo, valor, inf, sup, cor e destaque."""
    for y, linha in enumerate(linhas):
        ax.plot([linha["inf"], linha["sup"]], [y, y], color=linha["cor"], linewidth=2, zorder=2)
        ax.plot(linha["valor"], y, "o", color=linha["cor"], zorder=3)
        ax.annotate(
            f"{numero(linha['valor'], casas)}  [{numero(linha['inf'], casas)} ; {numero(linha['sup'], casas)}]",
            (linha["sup"], y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9,
            color=TINTA if linha.get("destaque") else TINTA_SECUNDARIA, fontweight="bold" if linha.get("destaque") else "normal",
        )
    ax.set_yticks(range(len(linhas)), [linha["rotulo"] for linha in linhas])
    ax.set_ylim(-0.6, len(linhas) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def auc_por_rotulo(tabela: pd.DataFrame, treinado: str) -> Figure:
    """A AUC do mesmo modelo conforme o gabarito."""
    with plt.rc_context(ESTILO):
        recorte = tabela.loc[treinado]
        menor, maior = recorte["AUC"].min(), recorte["AUC"].max()
        fig, ax = nova_figura(
            f"O mesmo modelo vale de {numero(menor, 3)} a {numero(maior, 3)} de AUC, conforme o rótulo usado como gabarito",
            'Modelo treinado com "não recomendaria", avaliado no teste contra cada definição de reclamação. Intervalo de 95% por bootstrap.',
            altura=3.4,
        )
        fig.subplots_adjust(left=0.37, bottom=0.19, right=0.97, top=0.76)
        linhas = [
            {"rotulo": f"{nome} (n = {int(linha['avaliações']):,d})".replace(",", "."), "valor": linha["AUC"], "inf": linha["IC 95% inferior"], "sup": linha["IC 95% superior"],
             "cor": AZUL if nome == "não recomendaria" else APAGADO, "destaque": nome == "não recomendaria"}
            for nome, linha in recorte.iterrows()
        ]
        _pontos_com_intervalo(ax, linhas)
        ax.set_xlim(menor - 0.006, maior + 0.019)
        ax.set_xticks(np.arange(np.floor(menor * 100) / 100, 1.0001, 0.01))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 2)))
        ax.set_xlabel("AUC no teste (1 = separa perfeitamente; 0,5 = acaso)")
        return fig


def calibracao(curvas: pd.DataFrame, resumo: dict) -> Figure:
    """Probabilidade prevista contra frequência observada, por décimo das avaliações."""
    with plt.rc_context(ESTILO):
        bruta, recalibrada = resumo["bruta"]["erro_de_calibracao"], resumo["recalibrada"]["erro_de_calibracao"]
        if recalibrada < bruta and resumo["diferenca_recalibrada_menos_bruta"]["superior"] < 0:
            titulo = f"Recalibrar reduz o erro de calibração de {numero(bruta, 3)} para {numero(recalibrada, 3)}"
        else:
            titulo = f"Recalibrar não melhora a calibração ({numero(bruta, 3)} → {numero(recalibrada, 3)})"
        fig, ax = nova_figura(
            titulo, "Cada ponto é um décimo das avaliações do teste: probabilidade média prevista\ne parcela que de fato era reclamação.", altura=5.0, largura=6.6,
        )
        fig.subplots_adjust(left=0.12, bottom=0.12, top=0.78)
        ax.plot([0, 1], [0, 1], color=EIXO, linewidth=1.2, zorder=1)
        ax.annotate("calibração perfeita", (0.62, 0.62), xytext=(8, -12), textcoords="offset points", color=APAGADO, fontsize=8.5)
        for versao, cor, nome in (("bruta", APAGADO, "probabilidade bruta do modelo"), ("recalibrada", AZUL, "recalibrada (Platt, ajustada em abril)")):
            dados = curvas[curvas["versão"] == versao]
            ax.plot(dados["prevista"], dados["observada"], "-o", color=cor, label=nome, zorder=3 if versao == "recalibrada" else 2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(_PCT)
        ax.yaxis.set_major_formatter(_PCT)
        ax.set_xlabel("probabilidade prevista de ser reclamação")
        ax.set_ylabel("parcela que era reclamação")
        ax.legend(loc="upper left", fontsize=9)
        return fig


def estabilidade_dos_temas(tabela: pd.DataFrame, escolhido: int, outros: dict[str, float] | None = None) -> Figure:
    """Estabilidade dos temas conforme o número de temas pedido."""
    with plt.rc_context(ESTILO):
        linha = tabela.loc[escolhido]
        fig, ax = nova_figura(
            f"Os temas mudam a cada execução: estabilidade de {numero(tabela['estabilidade'].min())} a {numero(tabela['estabilidade'].max())} "
            f"com {tabela.index.min()} a {tabela.index.max()} temas",
            "Estabilidade: Jaccard médio das 10 palavras do topo entre temas casados, em pares de execuções\n(outra semente, outra subamostra de 80% das reclamações). Faixa: do par menos parecido ao mais parecido.",
            altura=4.6,
        )
        fig.subplots_adjust(bottom=0.13, top=0.78)
        ax.fill_between(tabela.index, tabela["mínimo"], tabela["máximo"], color=AZUL, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(tabela.index, tabela["estabilidade"], "-o", color=AZUL, zorder=3)
        ax.annotate(
            f"NMF sobre TF-IDF, {escolhido} temas (o adotado): {numero(linha['estabilidade'])}", (escolhido, linha["estabilidade"]),
            xytext=(0, 34), textcoords="offset points", ha="center", color=TINTA, fontsize=9, fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": EIXO, "linewidth": 0.8},
        )
        if outros:
            for (nome, valor), cor, desvio in zip(outros.items(), (AQUA, AMARELO), (-0.12, 0.12)):  # desvio: os dois não se cobrem
                ax.plot(escolhido + desvio, valor, "D", color=cor, zorder=4, label=f"{nome}, {escolhido} grupos (amostra): {numero(valor)}")
            ax.legend(loc="lower left", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(list(tabela.index))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 1)))
        ax.set_xlabel("número de temas pedido ao NMF")
        ax.set_ylabel("estabilidade (1 = os mesmos temas sempre)")
        return fig


def motivos(tabela: pd.DataFrame) -> Figure:
    """Parcela das reclamações em cada motivo, com os dois intervalos."""
    with plt.rc_context(ESTILO):
        ordenada = tabela[(tabela["temas"] > 0) | (tabela["maior entre execuções"] > 0.005)].sort_values("parcela", ascending=False)
        amplitude = ordenada["reajuste sup"] - ordenada["reajuste inf"]
        instavel = amplitude.idxmax()
        altura = 0.44 * len(ordenada) + 2.1
        fig, ax = nova_figura(
            f"O tamanho de cada motivo muda ao reaprender os temas: de {percentual(ordenada.loc[instavel, 'reajuste inf'])} "
            f"a {percentual(ordenada.loc[instavel, 'reajuste sup'])} no caso extremo",
            "Parcela das reclamações de maio de 2018 por motivo dominante (temas aprendidos de janeiro a março). Traço grosso: intervalo de 95%\n"
            "só da amostragem. Traço fino: intervalo de 95% reaprendendo os temas com outra semente e outra subamostra.",
            altura=altura,
        )
        fig.subplots_adjust(left=0.34, right=0.97, bottom=0.62 / altura, top=1 - 1.15 / altura)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        for y, (nome, linha) in enumerate(ordenada.iterrows()):
            ax.barh(y, linha["parcela"], height=0.6, color=AZUL, zorder=2)
            ax.plot([linha["reajuste inf"], linha["reajuste sup"]], [y, y], color=TINTA, linewidth=1.0, zorder=3)
            ax.plot([linha["amostragem inf"], linha["amostragem sup"]], [y, y], color=TINTA, linewidth=3.0, zorder=4)
            fim = max(linha["reajuste sup"], linha["amostragem sup"], linha["parcela"])
            ax.annotate(
                f"{percentual(linha['parcela'])}  (de {percentual(linha['reajuste inf'])} a {percentual(linha['reajuste sup'])})",
                (fim, y), xytext=(7, 0), textcoords="offset points", va="center", color=TINTA_SECUNDARIA, fontsize=8.5,
            )
        ax.set_yticks(range(len(ordenada)), list(ordenada.index))
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(_PCT)
        ax.set_xlim(0, max(ordenada["reajuste sup"].max(), ordenada["parcela"].max()) * 1.32)
        ax.set_xlabel("parcela das reclamações (entre parênteses: intervalo reaprendendo os temas)")
        return fig


def motivos_por_categoria(tabela: pd.DataFrame, geral: pd.Series) -> Figure:
    """Mapa de calor: parcela de cada motivo dentro de cada categoria de produto."""
    with plt.rc_context(ESTILO):
        colunas = [c for c in geral.sort_values(ascending=False).index if geral[c] > 0.005]  # motivos com menos de 0,5% ficam fora do mapa
        valores = tabela[colunas]
        diferenca = (valores - geral[colunas]).abs()
        categoria, motivo = diferenca.stack().idxmax()
        altura = 0.42 * len(valores) + 2.5
        fig, ax = nova_figura(
            f"Em {categoria}, “{motivo}” é {percentual(valores.loc[categoria, motivo])} das reclamações; no conjunto, {percentual(geral[motivo])}",
            "Parcela das reclamações de cada categoria de produto por motivo dominante (abril e maio de 2018). Cada linha soma 100%.",
            altura=altura, largura=9.2,
        )
        fig.subplots_adjust(left=0.25, right=0.98, bottom=1.25 / altura, top=1 - 0.95 / altura)
        mapa = LinearSegmentedColormap.from_list("azul", [SUPERFICIE, *RAMPA_AZUL])
        maximo = float(valores.to_numpy().max())
        ax.imshow(valores.to_numpy(), cmap=mapa, vmin=0, vmax=maximo, aspect="auto")
        ax.grid(visible=False)
        for i in range(valores.shape[0]):
            for j in range(valores.shape[1]):
                v = valores.iat[i, j]
                ax.text(j, i, percentual(v), ha="center", va="center", fontsize=8.5, color="#ffffff" if v > 0.55 * maximo else TINTA)
        ax.set_xticks(range(len(colunas)), [c.replace(" ou ", "\nou ").replace(" e ", "\ne ") for c in colunas], rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(valores)), [f"{nome} (n = {n})" for nome, n in zip(valores.index, tabela["reclamações"])])
        ax.set_xticks(np.arange(-0.5, len(colunas)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(valores)), minor=True)
        ax.grid(which="minor", color=SUPERFICIE, linewidth=2)
        ax.tick_params(which="both", length=0)
        for lado in ax.spines.values():
            lado.set_visible(False)
        return fig
