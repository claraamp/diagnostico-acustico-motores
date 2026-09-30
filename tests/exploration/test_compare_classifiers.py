"""Critério de escolha do compare_classifiers: não depende da ordem da lista."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")   # o módulo importa o torch (requirements.txt, índice CPU)

_CAMINHO = Path(__file__).resolve().parents[2] / "scripts" / "exploration" / "compare_classifiers.py"
_spec = importlib.util.spec_from_file_location("compare_classifiers", _CAMINHO)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

CUSTOS = {m: cc.custo(m) for m in ("lda", "mlp", "cnn1d", "cnn2d")}


def _escolher(medias, **kw):
    return cc.escolher(medias, CUSTOS, kw.pop("margem", 0.02), **kw)[0]


def test_empate_fica_o_mais_barato_em_qualquer_ordem():
    # caso do exp222: LDA e CNN 2D empatam exatamente
    base = {"lda": 0.833, "mlp": 0.680, "cnn1d": 0.668, "cnn2d": 0.833}
    for ordem in (["lda", "mlp", "cnn1d", "cnn2d"], ["cnn2d", "cnn1d", "mlp", "lda"],
                  ["cnn2d", "lda"]):
        assert _escolher({m: base[m] for m in ordem}) == "lda"


def test_dentro_da_margem_fica_o_mais_barato():
    assert _escolher({"lda": 0.860, "cnn1d": 0.875}) == "lda"


def test_mais_caro_entra_se_ganhar_por_mais_que_a_margem():
    assert _escolher({"lda": 0.800, "cnn1d": 0.875}) == "cnn1d"


def test_dispersao_entre_sementes_alarga_a_tolerancia():
    medias = {"lda": 0.800, "cnn1d": 0.840}
    assert _escolher(medias) == "cnn1d"
    assert _escolher(medias, desvios={"lda": 0.0, "cnn1d": 0.05}) == "lda"