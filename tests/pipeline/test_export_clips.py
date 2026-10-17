"""
Testes do 09_export_clips.py (carregado por caminho, porque o nome começa com dígito).
Não dependem de data/: o teste do clipe (C3) lê o .h versionado e recalcula, só a
partir dele, as características e o escore de cada segmento.

Garantias: a seleção é por posição (o meio de cada gravação), contígua, sem olhar o
sinal; o .h volta igual do texto; os dados são definidos num único .c (com dois .c
incluindo o .h, o áudio entra uma vez só na flash); e, no .h versionado:
- cada segmento, recalculado do int16 com o dsp.mfcc e a LDA final, dá o escore e a
  decisão gravados nele;
- o recorte a 51,2 kHz, decimado como na placa (fase 0, descartando as primeiras
  saídas) e requantizado, é bit a bit o primeiro segmento do clipe da normal, e os
  dois escores de referência do recorte são os da saída em float e em int16;
- decimado em float32, como a CMSIS na placa (sem e com FMA), o recorte fica a no
  máximo 1 LSB do clipe e os escores, dentro da tolerância encadeada do A4;
- o .h é do lda_final.json atual e diz que os segmentos são do treino.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
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
H = C.H                              # o 06: compilador_c e CFLAGS

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


def _header_sintetico(com_recorte: bool = True) -> tuple[str, list, object]:
    p, _, _ = _modelo()
    rng = np.random.default_rng(3)
    clipes = []
    for i, rotulo in enumerate(["normal", "bpfi_0.3mm"]):
        segs = _gravacao(rotulo, 2, 10 * i)
        clipes.append(C.Clipe(rotulo, segs, rng.integers(-3000, 3000, 2 * N).astype(np.int16),
                              [-1234.5 + i, 987.25], [False, True]))
    recorte = None
    if com_recorte:
        spec = dsp.design_decimation(config.FS_ORIGINAL, config.FS_TRABALHO)
        recorte = C.Recorte(segmento=clipes[0].segmentos[0], pcm=rng.integers(-3000, 3000, 1000)
                            .astype(np.int16), inicio_original=0, antes=75, depois=73, descartar=37,
                            taps=dsp.taps_decimacao(spec, config.FS_ORIGINAL), escore_float=-1.5,
                            escore_int16=-1.25, difere_na_placa={"sem FMA": 0, "com FMA": 1})
    origem = {"json": "x.json", "sha256_json": "0" * 64, "csv": "x.csv", "sha256_csv": "1" * 64,
              "splits": "abc", "commit": "teste00"}
    return C.gerar_header(clipes, recorte, origem, p), clipes, recorte


def test_header_volta_igual_do_texto():
    texto, clipes, _ = _header_sintetico(com_recorte=False)
    lido = C.ler_header(texto)
    assert [c["rotulo"] for c in lido["clipes"]] == ["normal", "bpfi_0.3mm"]
    for c, l in zip(clipes, lido["clipes"]):
        np.testing.assert_array_equal(l["pcm"], c.pcm)
    assert [s["id"] for s in lido["segmentos"]] == [0, 1, 10, 11]
    assert [s["falha"] for s in lido["segmentos"]] == [False, True, False, True]
    assert lido["segmentos"][1]["escore"] == pytest.approx(987.25)
    assert "recorte" not in lido and texto.isascii()
    assert "TODOS OS SEGMENTOS ESTAO NO TREINO" in texto
    # o .h só declara; as definições ficam atrás do CLIPES_DEMO_DADOS, e nada é static
    assert "static const" not in texto.replace("com static const no .h", "")
    cabeca, dados = texto.split("#if defined(CLIPES_DEMO_DADOS)")
    assert "extern const int16_t clipe_normal[" in cabeca and "= {" not in cabeca
    assert "const int16_t clipe_normal[" in dados


_PROGRAMA = r"""
#include <stdio.h>
#include "clipes_demo.h"
int main(void) {
    long long s = 0, r = 0;
    double f = 0.0;
    for (int c = 0; c < CLIPES_N_CLIPES; c++)
        for (uint32_t i = 0; i < clipes[c].n_amostras; i++)
            s += (long long)clipes[c].pcm[i] * (long long)(i % 7 + 1);
    for (int i = 0; i < RECORTE_N_AMOSTRAS; i++)
        r += (long long)recorte_51200[i] * (long long)(i % 5 + 1);
    for (int i = 0; i < RECORTE_N_TAPS; i++)
        f += (double)recorte_fir[i];
    printf("%lld %lld %.17g %.9g %u\n", s, r, f,
           (double)clipes_segmentos[CLIPES_N_SEGMENTOS - 1].escore_ref,
           (unsigned)clipes_segmentos[1].falha_ref);
    return 0;
}
"""


@pytest.mark.skipif(H.compilador_c() is None, reason="sem compilador C no host")
def test_dados_definidos_num_unico_c(tmp_path):
    """Um .c define os dados (e pode incluir o .h duas vezes); outro só os usa; o C lê os valores do Python."""
    texto, clipes, recorte = _header_sintetico()
    (tmp_path / "clipes_demo.h").write_text(texto, encoding="ascii", newline="\n")
    (tmp_path / "dados.c").write_text('#define CLIPES_DEMO_DADOS\n#include "clipes_demo.h"\n'
                                      '#include "clipes_demo.h"\n', encoding="ascii")
    (tmp_path / "main.c").write_text(_PROGRAMA, encoding="ascii")
    exe = tmp_path / "clipes"
    r = subprocess.run([H.compilador_c(), *H.CFLAGS, "-o", str(exe), str(tmp_path / "dados.c"),
                        str(tmp_path / "main.c")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s, rr, f, escore, falha = subprocess.run([str(exe)], capture_output=True, text=True,
                                             check=True).stdout.split()
    pesos7 = lambda n: np.arange(n) % 7 + 1
    assert int(s) == sum(int(np.dot(c.pcm.astype(np.int64), pesos7(len(c.pcm)))) for c in clipes)
    assert int(rr) == int(np.dot(recorte.pcm.astype(np.int64), np.arange(len(recorte.pcm)) % 5 + 1))
    assert float(f) == pytest.approx(float(recorte.taps.astype(np.float32).astype(np.float64).sum()),
                                     rel=1e-12)
    assert float(escore) == pytest.approx(987.25) and falha == "1"


@pytest.mark.skipif(shutil.which("arm-none-eabi-gcc") is None, reason="sem arm-none-eabi-gcc")
def test_audio_entra_uma_vez_na_flash(tmp_path):
    """Para o Cortex-M4, sem otimização (a configuração Debug): só o .c dos dados leva o áudio."""
    texto, clipes, recorte = _header_sintetico()
    (tmp_path / "clipes_demo.h").write_text(texto, encoding="ascii", newline="\n")
    (tmp_path / "dados.c").write_text('#define CLIPES_DEMO_DADOS\n#include "clipes_demo.h"\n',
                                      encoding="ascii")
    (tmp_path / "uso.c").write_text('#include "clipes_demo.h"\n'
                                    "int amostra(int c, int i) { return clipes[c].pcm[i] + recorte_51200[i]; }\n",
                                    encoding="ascii")
    tamanho = {}
    for nome in ("dados", "uso"):
        r = subprocess.run(["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16",
                            "-mfloat-abi=hard", "-std=c99", "-O0", "-Wall", "-Wextra", "-Werror",
                            "-c", "-o", str(tmp_path / f"{nome}.o"), str(tmp_path / f"{nome}.c")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        saida = subprocess.run(["arm-none-eabi-size", str(tmp_path / f"{nome}.o")],
                               capture_output=True, text=True, check=True).stdout
        tamanho[nome] = int(saida.splitlines()[1].split()[0])          # text (inclui .rodata)
    audio = sum(c.pcm.nbytes for c in clipes) + recorte.pcm.nbytes
    assert tamanho["dados"] >= audio
    assert tamanho["uso"] < 1024, "o .h sem CLIPES_DEMO_DADOS não pode levar o áudio"


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
    np.testing.assert_array_equal(h["recorte_fir"], h["recorte_fir"][::-1])   # a CMSIS inverte
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


@pytest.mark.skipif(not VERSIONADO.exists(), reason="clipes_demo.h ainda não gerado")
def test_recorte_decimado_em_float32_como_a_cmsis():
    """Em float32 (placa) o recorte não precisa ser bit a bit: até 1 LSB e o escore na tolerância."""
    h = C.ler_header(VERSIONADO.read_text(encoding="ascii"))
    if "recorte" not in h:
        pytest.skip("clipes_demo.h sem o recorte")
    clipe = h["clipes"][int(h["defs"]["RECORTE_CLIPE"])]["pcm"][:N]
    _, pesos, bias = _modelo()
    ref_float = float(h["defs"]["RECORTE_ESCORE_REF"].rstrip("f"))
    ref_int16 = float(h["defs"]["RECORTE_ESCORE_REF_INT16"].rstrip("f"))
    tol = next(c for e, _, _, c in dsp.ESTAGIOS_PORTE if e == "escore")
    for fma in (False, True):
        y32 = C.decimar_cmsis_f32(h["recorte"], h["recorte_fir"], int(h["defs"]["RECORTE_DESCARTAR"]),
                                  fma=fma)
        q = C.requantizar_f32(y32)
        dif = np.abs(q.astype(np.int32) - clipe.astype(np.int32))
        assert dif.max() <= 1 and np.count_nonzero(dif) <= 10, f"fma={fma}"
        assert C.escore_lda(y32.astype(np.float64), pesos, bias) == pytest.approx(ref_float, rel=tol)
        assert C.escore_lda(q / config.INT16_FULL, pesos, bias) == pytest.approx(ref_int16, rel=tol)


def test_emulacao_da_cmsis_e_o_fir_causal():
    """Sem arredondamento relevante, a emulação da CMSIS é o lfilter de fase 0 (o mesmo decimador)."""
    rng = np.random.default_rng(5)
    x = rng.integers(-20000, 20000, 4 * 300).astype(np.int16)
    fir = rng.normal(size=11).astype(np.float32)
    fir = (fir + fir[::-1]) / 2
    ref = C.decimar_recorte(x, fir.astype(np.float64), 3, 4, 0)
    for fma in (False, True):
        np.testing.assert_allclose(C.decimar_cmsis_f32(x, fir, 3, 4, fma=fma), ref, rtol=0, atol=1e-5)
