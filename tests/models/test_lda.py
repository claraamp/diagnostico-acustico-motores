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


def _mfcc_pela_descricao(pcm: np.ndarray, c: dict) -> np.ndarray:
    """MFCC refeito só a partir da descrição do JSON, sem chamar o dsp (como o firmware fará)."""
    from scipy.fft import dct
    x = pcm.astype(float) / c["escala_entrada"]["divisor"]
    q, n_fft = c["quadros"], c["fft"]["n_fft"]
    n = q["amostras_por_quadro"]
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))          # Hann simétrica
    bins = c["mel"]["bins"]
    fb = np.zeros((c["mel"]["n_filtros"], n_fft // 2 + 1))
    for m in range(1, c["mel"]["n_filtros"] + 1):
        l, k, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:k] = (np.arange(l, k) - l) / (k - l)
        fb[m - 1, k:r] = (r - np.arange(k, r)) / (r - k)
    linhas = []
    for i in range(q["n_quadros"]):
        quadro = x[i * q["passo"]: i * q["passo"] + n] * w
        p = np.abs(np.fft.rfft(quadro, n=n_fft)) ** 2
        linhas.append(np.log(fb @ p + c["log"]["epsilon"]))
    return dct(np.array(linhas), type=2, axis=1, norm="ortho")[:, :c["dct"]["coeficientes"]]


def test_descricao_da_cadeia_reproduz_o_dsp():
    """Quem implementar o MFCC só pela descrição do JSON chega ao mesmo resultado do dsp."""
    import config
    import dsp
    c = lda.cadeia_entrada()
    assert not c["mel"]["bins_coincidentes"]
    pcm = np.random.default_rng(1).integers(-20000, 20000, c["amostras_por_segmento"]).astype(np.int16)
    ref = dsp.mfcc(pcm.astype(float) / config.INT16_FULL, c["fs_hz"])
    assert ref.shape == (c["quadros"]["n_quadros"], c["dct"]["coeficientes"])
    assert np.allclose(_mfcc_pela_descricao(pcm, c), ref, rtol=1e-9, atol=1e-9)


def test_origem_do_sinal():
    """A origem descrita é o projeto do 02 (147 taps, corte 5.760 Hz, fator 4, como em reports/decimation/)."""
    import config
    import dsp
    o = lda.cadeia_entrada()["origem_do_sinal"]
    fir = dsp.design_decimation(config.FS_ORIGINAL, config.FS_TRABALHO)
    assert (o["fir"]["numtaps"], o["fir"]["corte_hz"], o["fator_decimacao"]) == (fir.numtaps, fir.cutoff_hz, fir.down)
    assert (fir.numtaps, fir.cutoff_hz, fir.down) == (147, 5760.0, 4)
    assert "73 amostras" in o["descricao"]                          # atraso de grupo descartado
