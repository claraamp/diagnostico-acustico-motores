"""
Fim de linha: LF no repositório, no checkout e nos arquivos gerados.

Hashes calculados sobre os bytes (o do splits.json, que vai para o registry, e o
SHA-256 do lda_final.json gravado no lda_modelo.h) mudam com um CRLF a mais sem
nenhum valor mudar. No Windows isso acontece de dois jeitos: o Git entrega o
arquivo com CRLF no checkout (resolvido pelo .gitattributes) ou o `write_text`
grava CRLF ao gerar (resolvido com `newline="\\n"` nos geradores).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from validation import particao

RAIZ = Path(__file__).resolve().parents[1]

# o que pode ficar fora de LF no repositório (ver .gitattributes)
SEM_CONVERSAO = (".csv",)                 # gravados pelo módulo csv, com CRLF, e mantidos como estão
BINARIOS = (".png", ".pdf", ".bin", ".npz", ".npy", ".mat", ".f64", ".i16")


@pytest.mark.parametrize("relativo", ["reports/modelo_final/lda_final.json",
                                      "reports/modelo_final/lda_modelo.h"])
def test_arquivos_com_hash_saem_do_checkout_em_lf(relativo):
    assert b"\r" not in (RAIZ / relativo).read_bytes(), (
        f"{relativo} está com CRLF, e o hash dele deixa de bater. Confira o .gitattributes "
        "e refaça o checkout (git rm -q --cached -r . e git reset -q --hard, com a árvore limpa)")


def test_particao_e_gravada_em_lf(tmp_path):
    caminho = particao.salvar({"parametros": {"fs_hz": 12800}, "segmentos": []},
                              tmp_path / "splits.json")
    assert b"\r" not in caminho.read_bytes()


def test_arquivos_versionados_de_texto_estao_em_lf():
    """Pega CRLF e CR solto (que o Git trata como binário, `-text`) em qualquer arquivo de texto."""
    if shutil.which("git") is None:
        pytest.skip("git não encontrado")
    r = subprocess.run(["git", "ls-files", "--eol"], cwd=RAIZ, capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode != 0:
        pytest.skip("fora de um repositório git")
    errados = []
    for linha in r.stdout.splitlines():
        info, _, caminho = linha.partition("\t")
        eol_indice = info.split()[0]                  # i/lf, i/crlf, i/mixed, i/-text, i/none
        if caminho.lower().endswith(SEM_CONVERSAO + BINARIOS):
            continue
        if eol_indice not in ("i/lf", "i/none"):
            errados.append(f"{eol_indice:8} {caminho}")
    assert not errados, ("arquivos de texto fora de LF no repositório (CRLF, misto ou CR solto); "
                         "corrija e rode git add --renormalize .:\n  " + "\n  ".join(errados))
