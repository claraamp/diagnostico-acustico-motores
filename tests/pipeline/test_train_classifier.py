"""
Testes do 05_train_classifier.py (carregado por caminho, porque o nome começa com
dígito). Não dependem de data/: usam um .npz e um manifesto sintéticos.

Garantias: as features só são aceitas se vierem da mesma partição, do mesmo MFCC
do config e sem --norm-clipe; os escores de referência saem da forma dobrada.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import config
from models import lda

RAIZ = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "train_classifier", RAIZ / "scripts" / "pipeline" / "05_train_classifier.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

N = 20


def _features(tmp_path, **manifesto_extra):
    X = np.random.default_rng(0).normal(size=(N, 2 * config.MFCC_N_COEFS))
    np.savez(tmp_path / "mfcc_features.npz", X=X)
    manifesto = {"splits_hash": "abc", "fs_hz": config.FS_TRABALHO,
                 "mfcc_janela_ms": config.MFCC_WINDOW_MS, "mfcc_hop_ms": config.MFCC_HOP_MS,
                 "mfcc_n_mels": config.MFCC_N_MELS, "mfcc_n_coefs": config.MFCC_N_COEFS,
                 "norm_clipe": False, "git_commit": "c1", **manifesto_extra}
    (tmp_path / "manifest_features.json").write_text(json.dumps(manifesto), encoding="utf-8")
    return tmp_path / "mfcc_features.npz"


def test_aceita_features_de_referencia(tmp_path):
    X, manifesto = T.carregar_features(_features(tmp_path), "abc", N)
    assert X.shape == (N, 26) and manifesto["git_commit"] == "c1"


@pytest.mark.parametrize("hash_splits, n, extra", [
    ("outro", N, {}),                       # outra partição
    ("abc", N + 1, {}),                     # número de linhas diferente da partição
    ("abc", N, {"norm_clipe": True}),       # ablação da normalização
    ("abc", N, {"mfcc_n_mels": 40}),        # outro MFCC
])
def test_recusa_features_divergentes(tmp_path, hash_splits, n, extra):
    with pytest.raises(SystemExit):
        T.carregar_features(_features(tmp_path, **extra), hash_splits, n)


def test_escores_de_referencia(tmp_path):
    X, _ = T.carregar_features(_features(tmp_path), "abc", N)
    y = np.array(["falha"] * 15 + ["normal"] * 5)
    X[y == "falha"] += 3
    p = lda.parametros(lda.treinar(X, y), config.MFCC_N_COEFS)
    segmentos = [SimpleNamespace(id=i, rotulo="g", indice=i) for i in range(N)]
    T.gravar_escores(tmp_path / "escores.csv", segmentos, y, p, X)
    linhas = list(csv.DictReader((tmp_path / "escores.csv").open(encoding="utf-8")))
    assert len(linhas) == N
    for linha, e in zip(linhas, lda.escore_dobrado(p, X)):
        assert float(linha["escore"]) == pytest.approx(e, rel=1e-8)
        assert linha["previsao"] == ("falha" if e > 0 else "normal")