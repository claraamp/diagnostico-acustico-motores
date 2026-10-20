"""Critério de escolha do compare_classifiers: não depende da ordem da lista."""

import importlib.util
from pathlib import Path

import numpy as np
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

def test_svm_linear_custa_o_mesmo_que_a_lda():
    assert cc.custo("svm_lin") == cc.custo("lda")


def test_svm_linear_separa_e_compensa_o_desbalanceamento():
    # 60 falhas para 10 normais, como o treino do B (~177 para ~48): com pesos
    # balanceados, a normal deslocada não é engolida pela classe maior
    rng = np.random.default_rng(0)
    X_tr = np.vstack([rng.normal(0, 1, (10, 26)), rng.normal(0, 1, (60, 26)) + 3])
    y_tr = np.array([0] * 10 + [1] * 60)
    X_te = np.vstack([rng.normal(0, 1, (20, 26)), rng.normal(0, 1, (20, 26)) + 3])
    pred = cc.treinar_svm_lin(X_tr, y_tr, X_te, 2)
    assert (pred[:20] == 0).all() and (pred[20:] == 1).all()
