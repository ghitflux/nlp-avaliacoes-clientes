"""Baixa o corpus público B2W-Reviews01 para data/brutos/ (fora do git) e confere tamanho e hash.

Fonte: https://github.com/americanas-tech/b2w-reviews01 (licença CC BY-NC-SA 4.0, crédito à B2W Digital).
O arquivo tem cerca de 49 MB e não é versionado; nenhum texto de avaliação entra no repositório.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request

from nlp_avaliacoes import dados

URL = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"
AGENTE = {"User-Agent": "portfolio-estudos/1.0"}
BLOCO = 1 << 20


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    destino = dados.ARQUIVO_BRUTO
    destino.parent.mkdir(parents=True, exist_ok=True)

    if not destino.exists():
        print(f"Baixando {URL}")
        parcial = destino.with_suffix(".parcial")
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, headers=AGENTE), timeout=120) as resposta, parcial.open("wb") as saida:
                while bloco := resposta.read(BLOCO):
                    saida.write(bloco)
            parcial.replace(destino)
        finally:
            parcial.unlink(missing_ok=True)
    else:
        print(f"{destino} já existe; conferindo.")

    tamanho = destino.stat().st_size
    resumo = hashlib.sha256()
    with destino.open("rb") as entrada:
        while bloco := entrada.read(BLOCO):
            resumo.update(bloco)
    sha = resumo.hexdigest()
    print(f"Tamanho: {tamanho:,d} bytes".replace(",", "."))
    print(f"sha256:  {sha}")

    if tamanho != dados.TAMANHO_ESPERADO or sha != dados.SHA256_ESPERADO:
        raise SystemExit(
            "O arquivo baixado não confere com o usado no estudo "
            f"(esperado: {dados.TAMANHO_ESPERADO} bytes, sha256 {dados.SHA256_ESPERADO}). "
            "A fonte pode ter mudado: apague o arquivo e baixe de novo, ou revise os números do README."
        )
    print("Tamanho e hash conferem com o arquivo usado no estudo.")


if __name__ == "__main__":
    main()
