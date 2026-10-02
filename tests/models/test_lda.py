"""
Testes do modelo final (models/lda.py), com dados sintéticos; não dependem de data/.

Garantias: (1) os parâmetros exportados reproduzem o sklearn nas duas formas,
inclusive depois de passar por JSON; (2) escore ≥ 0 é falha, qualquer que seja a
ordem das classes no sklearn, e o empate exato segue o sklearn (falha); (3) o modelo é o mesmo `novo_modelo` do protocolo;
(4) o modelo final só aceita a tarefa binária.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from models import lda
from validation.run_protocol import novo_modelo


def _dados(semente=0, n_falha=236, n_normal=59, n_coefs=13):
    rng = np.random.default_rng(semente)
    escalas = rng.uniform(0.1, 50, 2 * n_coefs)              # colunas em escalas bem diferentes
    Xf = rng.normal(1.0, 1.0, (n_falha, 2 * n_coefs)) * escalas
    Xn = rng.normal(-1.0, 1.0, (n_normal, 2 * n_coefs)) * escalas
    X = np.vstack([Xf, Xn])
    y = np.array(["falha"] * n_falha + ["normal"] * n_normal)
    return X, y


def test_json_reproduz_o_sklearn():
    X, y = _dados()
    modelo = lda.treinar(X, y)
    p = json.loads(json.dumps(lda.parametros(modelo, 13)))      # ida e volta pelo JSON
    c = lda.conferir(modelo, p, X)
    assert c["previsoes_iguais"]
    assert c["max_dif_relativa"] < 1e-9
    assert np.allclose(lda.escore_dobrado(p, X), lda.escore_padronizado(p, X))
    assert np.array_equal(lda.prever(p, X), modelo.predict(X))


def test_escore_positivo_e_falha():
    X, y = _dados()
    p = lda.parametros(lda.treinar(X, y), 13)
    s = lda.escore_dobrado(p, X)
    assert s[y == "falha"].mean() > 0 > s[y == "normal"].mean()
    assert p["classe_positiva"] == "falha"


def test_mesmo_modelo_do_protocolo():
    X, y = _dados(semente=3)
    ref = novo_modelo("lda", 2).fit(X, y)
    p = lda.parametros(lda.treinar(X, y), 13)
    assert np.array_equal(lda.prever(p, X), ref.predict(X))
    assert p["priors"] == [0.5, 0.5]


def test_ordem_das_features():
    nomes = lda.nomes_features(13)
    assert nomes[0] == "media_c0" and nomes[12] == "media_c12" and nomes[13] == "desvio_c0"
    assert len(nomes) == 26


def test_recusa_multiclasse():
    X, y = _dados()
    y = y.copy()
    y[:10] = "bpfi_0.3mm"
    with pytest.raises(ValueError):
        lda.treinar(X, y)


def test_recusa_numero_de_coeficientes_errado():
    X, y = _dados()
    with pytest.raises(ValueError):
        lda.parametros(lda.treinar(X, y), 12)

def test_empate_segue_o_sklearn():
    """No binário, o sklearn só escolhe classes_[1] com decision_function > 0; o empate é classes_[0]."""
    X, y = _dados()
    modelo = lda.treinar(X, y)
    p = lda.parametros(modelo, 13)
    assert p["escore_zero"] == str(modelo[-1].classes_[0]) == "falha"
    assert p["escore"].startswith("escore >= 0")
    zerado = {**p, "dobrada": {"pesos": [0.0] * 26, "bias": 0.0}}   # escore exatamente 0
    assert lda.prever(zerado, X[:3]).tolist() == ["falha"] * 3