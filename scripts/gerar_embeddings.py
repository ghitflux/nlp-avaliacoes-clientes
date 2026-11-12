"""Calcula, uma vez, a representação densa de uma amostra fixa e guarda em data/brutos/ (fora do git).

Modelo: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, em ONNX quantizado, pela
biblioteca fastembed (sem PyTorch nem TensorFlow). É opcional: scripts/rodar_analise.py usa o
arquivo se ele existir e, se não existir, roda o resto e avisa.

    uv sync --group embeddings
    uv run --group embeddings python scripts/gerar_embeddings.py

Travas de orçamento (o script desiste se passar de uma delas): 500 MB de download do modelo e
1,5 GB de memória do processo. Lotes pequenos e poucas threads, para caber num notebook comum.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

from nlp_avaliacoes import dados

MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PASTA_DO_MODELO = dados.BRUTOS / "modelo_fastembed"
LIMITE_DE_DOWNLOAD_MB = 500
LIMITE_DE_MEMORIA_MB = 1500
PARCIAL = dados.BRUTOS / "embeddings_parcial.npz"
LOTE = 32
BLOCO = 1000  # a cada bloco, grava o parcial e confere a memória
THREADS = 2  # de propósito: a máquina fica utilizável enquanto o modelo roda (cerca de 20 minutos)
MAXIMO_DE_CARACTERES = 1000  # corta textos muito longos (o modelo lê no máximo 128 tokens de qualquer forma)


def pico_de_memoria_mb() -> float:
    """Pico de memória física do processo, em MB (Windows: psapi; outros sistemas: resource)."""
    if sys.platform == "win32":

        class Contadores(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (nome, ctypes.c_size_t)
                for nome in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        contadores = Contadores()
        contadores.cb = ctypes.sizeof(contadores)
        nucleo = ctypes.WinDLL("kernel32", use_last_error=True)
        nucleo.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Contadores), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(nucleo.GetCurrentProcess(), ctypes.byref(contadores), contadores.cb)
        return contadores.PeakWorkingSetSize / 2**20
    import resource

    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return pico / 2**20 if sys.platform == "darwin" else pico / 2**10


def tamanho_da_pasta_mb(pasta: Path) -> float:
    return sum(arquivo.stat().st_size for arquivo in pasta.rglob("*") if arquivo.is_file() and not arquivo.is_symlink()) / 2**20


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    PASTA_DO_MODELO.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(PASTA_DO_MODELO / "hf"))  # nada fora da pasta do projeto
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from fastembed import TextEmbedding  # importa depois de definir HF_HOME

    tabela, _ = dados.carregar()
    amostra = dados.amostra_para_densa(tabela)
    textos = amostra["original"].str.slice(0, MAXIMO_DE_CARACTERES).tolist()
    chaves = amostra["chave"].to_numpy().astype("U16")
    del tabela
    print(f"Amostra: {len(textos)} avaliações ({amostra['parte'].value_counts().to_dict()})")

    inicio = time.perf_counter()
    modelo = TextEmbedding(MODELO, cache_dir=str(PASTA_DO_MODELO), threads=THREADS)
    download = tamanho_da_pasta_mb(PASTA_DO_MODELO)
    print(f"Modelo em disco: {download:.0f} MB; memória após carregar: {pico_de_memoria_mb():.0f} MB")
    if download > LIMITE_DE_DOWNLOAD_MB:
        raise SystemExit(f"O modelo passou de {LIMITE_DE_DOWNLOAD_MB} MB: via densa abandonada.")

    # Trabalho em blocos, com um arquivo parcial: se o processo for interrompido, a próxima execução continua.
    vetores = np.zeros((len(textos), 384), dtype=np.float16)
    prontos = 0
    if PARCIAL.exists():
        with np.load(PARCIAL, allow_pickle=False) as parcial:
            if np.array_equal(parcial["chaves"], chaves):
                prontos = int(parcial["prontos"])
                vetores[:prontos] = parcial["vetores"][:prontos]
                print(f"Continuando de {prontos} avaliações já calculadas.")
    for comeco in range(prontos, len(textos), BLOCO):
        fim = min(comeco + BLOCO, len(textos))
        vetores[comeco:fim] = np.array(list(modelo.embed(textos[comeco:fim], batch_size=LOTE)), dtype=np.float16)
        np.savez(PARCIAL, chaves=chaves, vetores=vetores, prontos=np.array(fim))
        pico = pico_de_memoria_mb()
        print(f"  {fim} de {len(textos)}  {time.perf_counter() - inicio:.0f} s  pico de memória {pico:.0f} MB", flush=True)
        if pico > LIMITE_DE_MEMORIA_MB:
            raise SystemExit(f"O processo passou de {LIMITE_DE_MEMORIA_MB} MB de memória: via densa abandonada.")

    np.savez_compressed(
        dados.ARQUIVO_EMBEDDINGS, chaves=chaves, vetores=vetores, modelo=np.array(MODELO),
        segundos=np.array(round(time.perf_counter() - inicio)), pico_de_memoria_mb=np.array(round(pico_de_memoria_mb())),
        download_mb=np.array(round(download)), threads=np.array(THREADS),
    )
    PARCIAL.unlink(missing_ok=True)
    print(
        f"Gravado em {dados.ARQUIVO_EMBEDDINGS} ({dados.ARQUIVO_EMBEDDINGS.stat().st_size / 2**20:.1f} MB) em "
        f"{time.perf_counter() - inicio:.0f} s; pico de memória {pico_de_memoria_mb():.0f} MB"
    )


if __name__ == "__main__":
    main()
