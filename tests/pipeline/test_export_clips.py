"""
Testes do 09_export_clips.py (carregado por caminho, porque o nome começa com dígito).
Não dependem de data/: o teste do clipe (C3) lê o .h versionado e recalcula, só a
partir dele, as características e o escore de cada segmento.

Garantias: a seleção é por posição (o meio de cada gravação), contígua, sem olhar o
sinal; o .h volta igual do texto; e, no .h versionado:
- cada segmento, recalculado do int16 com o dsp.mfcc e a LDA final, dá o escore e a
  decisão gravados nele;
- o recorte a 51,2 kHz, decimado como na placa (fase 0, descartando as primeiras
  saídas) e requantizado, é bit a bit o primeiro segmento do clipe da normal, e os
  dois escores de referência do recorte são os da saída em float e em int16;
- o .h é do lda_final.json atual e diz que os segmentos são do treino.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import config
import dsp
import pcm_io

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / "scripts" / "pipeline" / "09_export_clips.py"
_spec = importlib.util.spec_from_file_location("export_clips", SCRIPT)
C = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = C          # o script usa dataclasses
_spec.loader.exec_module(C)

MODELO_FINAL = RAIZ / "reports" / "modelo_final"
VERSIONADO = RAIZ / C.OUT_VERSIONADO
N = config.amostras_por_segmento(config.FS_TRABALHO)


@dataclass
class _Seg:
    id: int
    rotulo: str
    inicio: int
    fim: int


def _gravacao(rotulo, n_seg, id0):
    return [_Seg(id0 + i, rotulo, i * N, (i + 1) * N) for i in range(n_seg)]


def _modelo():
    p = json.loads((MODELO_FINAL / "lda_final.json").read_text(encoding="utf-8"))
    return p, np.asarray(p["dobrada"]["pesos"], dtype=float), float(p["dobrada"]["bias"])


def test_selecao_e_o_meio_de_cada_gravacao_e_contigua():
    segs = _gravacao("a", 7, 0) + _gravacao("b", 4, 7)
    escolha = C.selecionar(segs, 2)
    assert [s.id for s in escolha["a"]] == [3, 4] and [s.id for s in escolha["b"]] == [9, 10]
    with pytest.raises(ValueError):
        C.selecionar(segs, 5)                          # "b" não tem 5 a partir do meio
    buraco = _gravacao("c", 4, 0)
    buraco[3] = _Seg(3, "c", 4 * N + 1, 5 * N + 1)
    with pytest.raises(ValueError):
        C.selecionar(buraco, 2)                        # segmentos não contíguos


def test_header_volta_igual_do_texto():
    p, pesos, bias = _modelo()
    rng = np.random.default_rng(3)
    clipes = []
    for i, rotulo in enumerate(["normal", "bpfi_0.3mm"]):
        segs = _gravacao(rotulo, 2, 10 * i)
        clipes.append(C.Clipe(rotulo, segs, rng.integers(-3000, 3000, 2 * N).astype(np.int16),
                              [-1234.5 + i, 987.25], [False, True]))
    origem = {"json": "x.json", "sha256_json": "0" * 64, "csv": "x.csv", "sha256_csv": "1" * 64,
              "splits": "abc", "commit": "teste00"}
    texto = C.gerar_header(clipes, None, origem, p)
    lido = C.ler_header(texto)
    assert [c["rotulo"] for c in lido["clipes"]] == ["normal", "bpfi_0.3mm"]
    for c, l in zip(clipes, lido["clipes"]):
        np.testing.assert_array_equal(l["pcm"], c.pcm)
    assert [s["id"] for s in lido["segmentos"]] == [0, 1, 10, 11]
    assert [s["falha"] for s in lido["segmentos"]] == [False, True, False, True]
    assert lido["segmentos"][1]["escore"] == pytest.approx(987.25)
    assert "recorte" not in lido and texto.isascii()
    assert "TODOS OS SEGMENTOS ESTAO NO TREINO" in texto


@pytest.mark.skipif(not VERSIONADO.exists(), reason="clipes_demo.h ainda não gerado")
def test_clipe_recalculado_em_python_da_o_escore_gravado():
    """C3: as características recalculadas do .h reproduzem o escore e a decisão de cada segmento."""
    texto = VERSIONADO.read_text(encoding="ascii")
    h = C.ler_header(texto)
    p, pesos, bias = _modelo()
    assert h["defs"]["CLIPES_JSON_SHA256_12"].strip('"') == \
        hashlib.sha256((MODELO_FINAL / "lda_final.json").read_bytes()).hexdigest()[:12], \
        "clipes_demo.h é de outro modelo: rode o 09 de novo"
    assert "TODOS OS SEGMENTOS ESTAO NO TREINO" in texto
    assert len(h["segmentos"]) == int(h["defs"]["CLIPES_N_SEGMENTOS"])
    for i, c in enumerate(h["clipes"]):
        assert len(c["pcm"]) == c["n_seg"] * N
        for k in range(c["n_seg"]):
            seg = h["segmentos"][c["primeiro"] + k]
            assert seg["clipe"] == i
            x = c["pcm"][k * N:(k + 1) * N].astype(np.float64) / config.INT16_FULL
            escore = C.escore_lda(x, pesos, bias)
            assert escore == pytest.approx(seg["escore"], rel=1e-6), f"segmento {seg['id']}"
            assert (escore >= 0) == seg["falha"], f"segmento {seg['id']}"


@pytest.mark.skipif(not VERSIONADO.exists(), reason="clipes_demo.h ainda não gerado")
def test_recorte_decimado_como_na_placa_e_o_clipe_a_12k8():
    h = C.ler_header(VERSIONADO.read_text(encoding="ascii"))
    if "recorte" not in h:
        pytest.skip("clipes_demo.h sem o recorte")
    d = {k: int(h["defs"][k]) for k in ("RECORTE_MARGEM_ANTES", "RECORTE_MARGEM_DEPOIS",
                                        "RECORTE_DESCARTAR", "RECORTE_FASE", "RECORTE_FATOR",
                                        "RECORTE_N_AMOSTRAS", "RECORTE_CLIPE", "RECORTE_SEGMENTO")}
    spec = dsp.design_decimation(config.FS_ORIGINAL, config.FS_TRABALHO)
    taps = dsp.taps_decimacao(spec, config.FS_ORIGINAL)
    # o FIR do .h é o do Python em float32, e as margens são as do dsp.recorte_para_decimar
    np.testing.assert_array_equal(h["recorte_fir"], taps.astype(np.float32))
    a, b, descartar = dsp.recorte_para_decimar(10 * N, N, spec, d["RECORTE_FASE"])
    assert (10 * N * d["RECORTE_FATOR"] - a, b - a, descartar) == \
        (d["RECORTE_MARGEM_ANTES"], d["RECORTE_N_AMOSTRAS"], d["RECORTE_DESCARTAR"])
    assert d["RECORTE_N_AMOSTRAS"] == d["RECORTE_FATOR"] * N + d["RECORTE_MARGEM_ANTES"] \
        + d["RECORTE_MARGEM_DEPOIS"]
    # decimado como na placa e requantizado: bit a bit o 1º segmento do clipe
    y = C.decimar_recorte(h["recorte"], taps, d["RECORTE_DESCARTAR"], d["RECORTE_FATOR"],
                          d["RECORTE_FASE"])
    clipe = h["clipes"][d["RECORTE_CLIPE"]]
    seg = h["segmentos"][clipe["primeiro"]]
    assert seg["id"] == d["RECORTE_SEGMENTO"]
    np.testing.assert_array_equal(pcm_io.para_pcm(y), clipe["pcm"][:N])
    # os dois escores do recorte: saída em float e saída requantizada (= o do clipe)
    _, pesos, bias = _modelo()
    ref_float = float(h["defs"]["RECORTE_ESCORE_REF"].rstrip("f"))
    ref_int16 = float(h["defs"]["RECORTE_ESCORE_REF_INT16"].rstrip("f"))
    assert C.escore_lda(y, pesos, bias) == pytest.approx(ref_float, rel=1e-6)
    assert ref_int16 == pytest.approx(seg["escore"], rel=1e-7)
    assert ref_float != ref_int16                    # a requantização muda o escore
