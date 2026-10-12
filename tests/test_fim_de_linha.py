"""
Os arquivos cujo hash é conferido ficam em LF, no repositório e quando gerados.

O hash é calculado sobre os bytes: um CRLF a mais muda o hash sem mudar nenhum
valor. Isso acontece no Windows de dois jeitos: o Git entrega o arquivo com
CRLF no checkout (resolvido pelo .gitattributes) ou o `write_text` grava CRLF
ao gerar (resolvido com `newline="\\n"` nos scripts que geram esses arquivos).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validation import particao

RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relativo", ["reports/modelo_final/lda_final.json",
                                      "reports/modelo_final/lda_modelo.h"])
def test_arquivos_com_hash_saem_do_checkout_em_lf(relativo):
    assert b"\r" not in (RAIZ / relativo).read_bytes(), (
        f"{relativo} está com CRLF, e o hash dele deixa de bater. Confira o .gitattributes "
        "e refaça o checkout (git rm --cached -r . e git reset --hard, com a árvore limpa)")


def test_particao_e_gravada_em_lf(tmp_path):
    caminho = particao.salvar({"parametros": {"fs_hz": 12800}, "segmentos": []},
                              tmp_path / "splits.json")
    assert b"\r" not in caminho.read_bytes()
