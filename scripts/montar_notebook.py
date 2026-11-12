"""Monta analise.ipynb com nbformat. Depois de montar, execute:

    uv run jupyter nbconvert --to notebook --execute --inplace analise.ipynb

O notebook não mostra texto de avaliação: só tabelas agregadas e figuras.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELULAS: list[tuple[str, str]] = [
    ("md", """# Do que os clientes reclamam, e dá para confiar numa leitura automática das avaliações?

Quem cuida de marketing e de produto recebe milhares de avaliações em texto livre e quer duas coisas delas: um
**alerta confiável de insatisfação** e uma **lista dos motivos de reclamação**, com o tamanho de cada um.

Este notebook conta a história com as tabelas. Tudo o que aparece aqui sai do pacote `nlp_avaliacoes`
(`src/`), e os mesmos números estão em `reports/resultados.json`. Base: B2W-Reviews01, avaliações de produtos
de um e-commerce brasileiro, de janeiro a maio de 2018 (licença CC BY-NC-SA 4.0). Nenhum texto de avaliação é
mostrado: só agregados."""),
    ("code", """%matplotlib inline
import warnings

import pandas as pd
from IPython.display import display

from nlp_avaliacoes import dados, graficos, pipeline, resumo_site

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
pd.set_option("display.max_colwidth", 110)

tabela, contagem = dados.carregar()
r = pipeline.executar(tabela, contagem, falar=False)  # a análise inteira: cerca de 3 minutos
print(f"{len(tabela)} avaliações usadas; MiniLM disponível: {r.densa_disponivel}")"""),
    ("md", """## 1. A base, a limpeza e a separação no tempo

Saem as avaliações sem resposta em "recomendaria", as de texto vazio, as repetidas e os textos copiados
(o mesmo texto em avaliações diferentes: fica a primeira no tempo, para o mesmo texto não cair no treino e no
teste). O modelo aprende de janeiro a março, é ajustado em abril e é testado em maio."""),
    ("code", """display(pd.Series({k: v for k, v in r.base.items() if not isinstance(v, dict)}, name="valor").to_frame())
display(pd.DataFrame(r.base["periodos"]).T)"""),
    ("md", """## 2. Dois rótulos que nem sempre concordam

Cada avaliação traz uma nota de 1 a 5 e a resposta a "recomendaria este produto?". Reclamação pode ser "nota
até 2" ou "não recomendaria". A tabela mostra, por nota, que parcela não recomendaria."""),
    ("code", """display(r.discordancia_por_nota)
display(pd.DataFrame(r.discordancia).T)"""),
    ("md", """## 3. O alerta: léxico, TF-IDF de palavras e TF-IDF de caracteres

Rótulo: "não recomendaria". A linha de base é uma lista de palavras negativas escrita à mão, sem treino.
Métricas no teste (maio), com intervalo de 95% por bootstrap; as diferenças são pareadas (mesmas avaliações
sorteadas para os dois métodos)."""),
    ("code", """display(r.desempenho.T)
display(r.diferencas.T)
for nome, grade in r.regularizacao.items():
    print(nome)
    display(grade)"""),
    ("code", """_ = graficos.curva_de_captura(r.curvas, pipeline.PALAVRAS, pipeline.LEXICO, 0.10)"""),
    ("md", """## 4. E a nota, que já vem pronta?

A regra "nota até 2" não lê texto nenhum. Contra o rótulo "não recomendaria", ela é comparada com o modelo de
texto marcando o mesmo número de avaliações. Depois, um modelo pequeno de segundo estágio mede quanto o texto
acrescenta a quem já tem a nota, e a última tabela mostra onde o texto ajuda: dentro de cada nota."""),
    ("code", """display(r.regra_da_nota.T)
display(r.nota_e_texto.T)
display(r.nota_e_texto_diferenca.T)
display(r.texto_dentro_da_nota)"""),
    ("md", """## 5. Ruído de rótulo: a métrica depende do gabarito

O mesmo escore, avaliado contra cada definição de reclamação. Treinar com um rótulo ou com o outro quase não
muda o modelo; trocar o gabarito muda a AUC."""),
    ("code", """display(r.auc_por_rotulo)
_ = graficos.auc_por_rotulo(r.auc_por_rotulo, '"não recomendaria"')"""),
    ("md", """Entre as avaliações de nota 4 ou 5 que recomendariam, quantas o modelo lê como reclamação? E o que há
nelas? A segunda tabela resume a leitura manual dos 45 "alarmes falsos" de maior escore e das 45 "reclamações
perdidas" de menor escore (`data/leitura_manual_de_erros.csv`, sem texto)."""),
    ("code", """display(pd.Series(r.nota_alta_texto_negativo, name="nota 4 ou 5, recomendaria").to_frame())
display(r.leitura)"""),
    ("md", """## 6. Calibração e onde o alerta erra"""),
    ("code", """display(pd.DataFrame({k: v for k, v in r.calibracao.items() if isinstance(v, dict)}).T)
_ = graficos.calibracao(r.curvas_de_calibracao, r.calibracao)"""),
    ("code", """display(r.erros)"""),
    ("md", """## 7. Motivos de reclamação: os temas e a sua estabilidade

Reclamações com rótulo sem ambiguidade (nota até 2 e "não recomendaria"). O NMF sobre TF-IDF é ajustado nas
reclamações de janeiro a março; o tamanho de cada motivo é medido em maio. A estabilidade responde: se eu
repetir o ajuste com outra semente e outra subamostra, os mesmos temas voltam?"""),
    ("code", """display(r.estabilidade)
display(pd.Series(r.estabilidade_do_modelo, name="modelo adotado").to_frame())
_ = graficos.estabilidade_dos_temas(r.estabilidade, r.numero_de_temas, resumo_site.estabilidade_dos_agrupamentos(r))"""),
    ("code", """display(r.temas)"""),
    ("md", """Cada tema recebe o nome de um motivo por uma regra de palavras-âncora (`analise.MOTIVOS`), aplicada da mesma
forma a qualquer execução. Dois intervalos: o de amostragem (temas fixos) e o de reajuste (temas reaprendidos)."""),
    ("code", """display(r.motivos)
display(r.familias)
_ = graficos.motivos(r.motivos)"""),
    ("code", """display(r.por_categoria)
_ = graficos.motivos_por_categoria(r.por_categoria, r.motivos["parcela"])"""),
    ("code", """display(r.por_mes)"""),
    ("md", """## 8. Representação densa contra TF-IDF, na mesma amostra

O modelo de linguagem (MiniLM multilíngue, em ONNX) foi rodado uma vez numa amostra fixa de 20 mil avaliações.
Classificadores treinados nas mesmas 9 mil avaliações e testados nas mesmas 5 mil; depois, agrupamento k-means
sobre a representação densa contra o NMF, nas mesmas reclamações."""),
    ("code", """display(pd.Series({k: v for k, v in r.amostra.items() if not isinstance(v, dict)}, name="amostra").to_frame())
display(r.representacoes.T)
display(r.representacoes_diferencas.T)"""),
    ("code", """display(r.agrupamentos)
display(r.agrupamentos_parcelas)"""),
    ("md", """## 9. O que fica

- **O alerta é confiável e simples.** TF-IDF de palavras com regressão logística; caracteres e representação
  densa não melhoram. O léxico feito à mão já acerta bastante na pilha pequena e perde quando é preciso ler mais.
- **Se a nota existe, ela é o melhor alerta.** O texto acrescenta pouco à nota no geral e ajuda mais na nota 3.
- **A métrica depende do rótulo.** Boa parte dos "erros" mais confiantes do modelo são erros do rótulo:
  reclamações de entrega escritas com nota alta.
- **Os motivos são um mapa, não uma medida.** Os temas mudam entre execuções e entre métodos; as famílias
  (entrega e loja, produto) são mais estáveis do que os motivos dentro delas."""),
]


def main() -> None:
    caderno = new_notebook()
    caderno.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    caderno.metadata["language_info"] = {"name": "python"}
    for tipo, conteudo in CELULAS:
        caderno.cells.append(new_markdown_cell(conteudo) if tipo == "md" else new_code_cell(conteudo))
    destino = Path("analise.ipynb")
    nbformat.write(caderno, destino)
    print(f"{destino} montado com {len(caderno.cells)} células. Execute com nbconvert (ver o topo deste arquivo).")


if __name__ == "__main__":
    main()
