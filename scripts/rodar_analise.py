"""Roda a análise completa e grava reports/resultados.json e as figuras do README.

Usa data/brutos/embeddings_minilm.npz se existir (gerado uma vez por scripts/gerar_embeddings.py);
sem ele, roda tudo o que não depende do modelo de linguagem e avisa.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from nlp_avaliacoes import dados, graficos, pipeline, resumo_site

SAIDA = Path("reports")
FIGURAS = SAIDA / "figuras"


def _tabela(tabela: pd.DataFrame, orient: str = "index"):
    """DataFrame como estrutura do JSON; índice de vários níveis vira lista de registros."""
    if isinstance(tabela.index, pd.MultiIndex) or orient == "records":
        return json.loads(tabela.reset_index().to_json(orient="records", force_ascii=False, double_precision=6))
    return json.loads(tabela.to_json(orient="index", force_ascii=False, double_precision=6))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIGURAS.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()

    tabela, contagem = dados.carregar()
    print(f"Base limpa: {len(tabela)} avaliações ({time.perf_counter() - inicio:.1f} s)")
    r = pipeline.executar(tabela, contagem)

    figuras = {
        "curva_de_captura": graficos.curva_de_captura(r.curvas, pipeline.PALAVRAS, pipeline.LEXICO, 0.10),
        "auc_por_rotulo": graficos.auc_por_rotulo(r.auc_por_rotulo, '"não recomendaria"'),
        "calibracao": graficos.calibracao(r.curvas_de_calibracao, r.calibracao),
        "estabilidade_dos_temas": graficos.estabilidade_dos_temas(r.estabilidade, r.numero_de_temas, resumo_site.estabilidade_dos_agrupamentos(r)),
        "motivos": graficos.motivos(r.motivos),
        "motivos_por_categoria": graficos.motivos_por_categoria(r.por_categoria, r.motivos["parcela"]),
    }
    for nome, figura in figuras.items():
        figura.savefig(FIGURAS / f"{nome}.png")
        plt.close(figura)

    resultados = {
        "base": r.base,
        "discordancia_por_nota": _tabela(r.discordancia_por_nota),
        "discordancia": r.discordancia,
        "regularizacao": {nome: _tabela(grade) for nome, grade in r.regularizacao.items()},
        "modelo_principal": r.principal,
        "melhor_modelo_na_validacao": r.melhor_na_validacao,
        "desempenho": _tabela(r.desempenho),
        "diferencas": _tabela(r.diferencas),
        "curvas_de_captura": _tabela(r.curvas, "records"),
        "regra_da_nota": _tabela(r.regra_da_nota),
        "auc_por_rotulo": _tabela(r.auc_por_rotulo),
        "texto_dentro_da_nota": _tabela(r.texto_dentro_da_nota),
        "nota_e_texto": _tabela(r.nota_e_texto),
        "nota_e_texto_diferenca": _tabela(r.nota_e_texto_diferenca),
        "nota_alta_texto_negativo": r.nota_alta_texto_negativo,
        "calibracao": r.calibracao,
        "curvas_de_calibracao": _tabela(r.curvas_de_calibracao, "records"),
        "erros_por_recorte": _tabela(r.erros),
        "leitura_manual_de_erros": _tabela(r.leitura),
        "leitura_manual_no_modelo_final": r.leitura_no_modelo_final,
        "estabilidade_por_numero_de_temas": _tabela(r.estabilidade),
        "numero_de_temas": r.numero_de_temas,
        "estabilidade_do_modelo_de_temas": r.estabilidade_do_modelo,
        "temas": _tabela(r.temas),
        "motivos": {"execucoes_do_reajuste": r.motivos.attrs["execucoes"], "tabela": _tabela(r.motivos)},
        "familias_de_motivos": _tabela(r.familias),
        "motivos_por_categoria": _tabela(r.por_categoria),
        "motivos_por_mes": _tabela(r.por_mes),
        "amostra_da_representacao_densa": {**r.amostra, "minilm_disponivel": r.densa_disponivel, "como_foi_gerado": dados.metadados_dos_embeddings()},
        "representacoes_na_amostra": _tabela(r.representacoes),
        "representacoes_na_amostra_diferencas": _tabela(r.representacoes_diferencas),
        "agrupamentos_na_amostra": {"reclamacoes": r.agrupamentos.attrs["reclamacoes"], "tabela": _tabela(r.agrupamentos)},
        "motivos_por_metodo_de_agrupamento": _tabela(r.agrupamentos_parcelas),
        "achados": resumo_site.achados(r),
        "para_o_site": resumo_site.para_o_site(r),
        "tempos_em_segundos": {**r.tempos, "total": round(time.perf_counter() - inicio)},
    }
    resumo_site.conferir(resultados["para_o_site"])
    (SAIDA / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 90)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    for titulo, conteudo in (
        ("Desempenho do alerta (teste)", r.desempenho), ("Diferenças pareadas", r.diferencas), ("Regra da nota", r.regra_da_nota),
        ("AUC por rótulo", r.auc_por_rotulo), ("Texto dentro da nota", r.texto_dentro_da_nota), ("Nota e texto", r.nota_e_texto),
        ("Erros por recorte", r.erros), ("Leitura manual", r.leitura), ("Estabilidade por número de temas", r.estabilidade),
        ("Temas", r.temas), ("Motivos", r.motivos), ("Famílias", r.familias), ("Por categoria", r.por_categoria), ("Por mês", r.por_mes),
        ("Representações na amostra", r.representacoes), ("Diferenças na amostra", r.representacoes_diferencas),
        ("Agrupamentos", r.agrupamentos), ("Motivos por método", r.agrupamentos_parcelas),
    ):
        print(f"\n== {titulo} ==\n{conteudo.to_string()}")
    print("\ncalibração:", r.calibracao)
    print("discordância:", r.discordancia)
    print("nota alta, texto negativo:", r.nota_alta_texto_negativo)
    print(f"\nGravado em {SAIDA / 'resultados.json'} e {FIGURAS}. Tempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
