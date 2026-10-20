"""
Testes do 10_prepare_hustmotor.py (carregado por caminho, porque o nome começa com
dígito). Não dependem de data/: escrevem um .txt sintético no formato do HUSTmotor.

Garantias: o leitor confere canais, blocos e o tempo que volta a zero em cada
bloco; cada bloco vira exatamente um segmento de 1 s, decimado sozinho (o
segmento k só tem o conteúdo do bloco k); a decimação preserva o que está abaixo
do corte e rejeita o que dobraria para dentro da banda; o .bin só se corta em
segmentos inteiros; a divisão do C2 cobre os 5 blocos e a falha mede com todos;
uma falha acima do pico da normal para, em vez de dar a volta no int16; e as
razões do 6203 são
as da fórmula.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import config

_CAMINHO = Path(__file__).resolve().parents[2] / "scripts" / "pipeline" / "10_prepare_hustmotor.py"
_spec = importlib.util.spec_from_file_location("prepare_hustmotor", _CAMINHO)
H = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = H
_spec.loader.exec_module(H)

FS = H.FS_HUST
N_BLOCO = 32768
N_BLOCOS = 5


def _escrever(caminho: Path, som: np.ndarray, legenda=H.CANAIS, tempo_continuo=False) -> Path:
    """Um .txt no formato do HUSTmotor: cabeçalho, tempo e 4 canais (só o som importa)."""
    n_total = som.size
    t = (np.arange(n_total) if tempo_continuo else np.tile(np.arange(N_BLOCO), N_BLOCOS)) / FS
    cab = ["Title:\tteste", "Parameters:", "", "DAQ Settings:", "Frequency Limit\t10000.000000",
           "Spectral Lines\t12800", f"Number of Blocks\t{N_BLOCOS}", f"Total Data Rows\t{n_total}",
           "Channels:", "Legend\t" + "\t".join(legenda), "Tacho1", "", "On/Off\tON\tON\tON\tON",
           "OFF", "", "Volts/Unit\t1.000000\t1.000000\t1.000000\t1.000000", "",
           "Time (seconds) and Data Channels"]
    zeros = np.zeros(n_total)
    corpo = np.column_stack([t, zeros, zeros, zeros, som])
    with caminho.open("w", encoding="latin-1", newline="\r\n") as f:
        f.write("\n".join(cab) + "\n")
        np.savetxt(f, corpo, fmt="%.8f", delimiter="\t")
    return caminho


def _tons(freqs) -> np.ndarray:
    """Um tom por bloco, cada bloco começando do zero (como aquisições separadas)."""
    t = np.arange(N_BLOCO) / FS
    return np.concatenate([0.05 * np.sin(2 * np.pi * f * t) for f in freqs])


def test_le_os_blocos_do_canal_de_som(tmp_path):
    som = _tons([300, 400, 500, 600, 700])
    b = H.ler_txt(_escrever(tmp_path / "a.txt", som))
    assert b.shape == (N_BLOCOS, N_BLOCO)
    np.testing.assert_allclose(b.ravel(), som, atol=1e-8)


def test_recusa_canais_errados_e_tempo_continuo(tmp_path):
    som = _tons([300] * N_BLOCOS)
    with pytest.raises(ValueError, match="canais"):
        H.ler_txt(_escrever(tmp_path / "a.txt", som, legenda=("X", "Y", "Sound", "Z")))
    with pytest.raises(ValueError, match="volta a zero"):
        H.ler_txt(_escrever(tmp_path / "b.txt", som, tempo_continuo=True))


def test_cada_bloco_vira_um_segmento_so_dele():
    freqs = [300.0, 700.0, 1100.0, 1700.0, 2300.0]
    blocos = {"x": _tons(freqs).reshape(N_BLOCOS, N_BLOCO)}
    sinais, spec = H.preparar(blocos, config.FS_TRABALHO, escala=0.98 * config.INT16_FULL / 0.05)
    n = config.amostras_por_segmento(config.FS_TRABALHO)
    assert spec.down == 2 and sinais["x"].size == N_BLOCOS * n
    for k, f0 in enumerate(freqs):
        seg = sinais["x"][k * n:(k + 1) * n]
        f = np.fft.rfftfreq(n, 1 / config.FS_TRABALHO)
        assert f[np.abs(np.fft.rfft(seg * np.hanning(n))).argmax()] == pytest.approx(f0, abs=1.0)


def test_decimacao_preserva_a_banda_e_rejeita_o_que_dobraria():
    t = np.arange(N_BLOCO) / FS
    escala = 0.98 * config.INT16_FULL / 0.05
    n = config.amostras_por_segmento(config.FS_TRABALHO)
    dentro = H.preparar({"x": np.tile(0.05 * np.sin(2 * np.pi * 1000 * t), (N_BLOCOS, 1))},
                        config.FS_TRABALHO, escala)[0]["x"][:n]
    # 9 kHz a 25,6 kHz dobraria para 3,8 kHz a 12,8 kHz sem o filtro
    fora = H.preparar({"x": np.tile(0.05 * np.sin(2 * np.pi * 9000 * t), (N_BLOCOS, 1))},
                      config.FS_TRABALHO, escala)[0]["x"][:n]
    rms = lambda v: np.sqrt(np.mean(v ** 2))
    assert rms(dentro) == pytest.approx(0.98 / np.sqrt(2), rel=0.01)
    assert 20 * np.log10(rms(fora) / rms(dentro)) < -55


def test_razoes_do_6203():
    assert H.RAZAO_BPFO == pytest.approx(3.0531, abs=1e-4)
    assert H.RAZAO_BPFI == pytest.approx(4.9469, abs=1e-4)
    assert H.RAZAO_BPFO + H.RAZAO_BPFI == pytest.approx(H.N_ESFERAS)


def test_corte_em_segmentos_independentes_e_divisao_do_c2():
    n = config.amostras_por_segmento(config.FS_TRABALHO)
    x = np.arange(N_BLOCOS * n, dtype=float)
    segs = H.cortar_segmentos(x, n, N_BLOCOS)
    assert segs.shape == (N_BLOCOS, n) and segs[1, 0] == n
    with pytest.raises(ValueError, match="não são"):
        H.cortar_segmentos(x[:-1], n, N_BLOCOS)
    # a divisão cobre os 5 blocos, sem repetir nenhum
    blocos = H.BLOCOS_CALIBRACAO + H.BLOCOS_DESCARTE + H.BLOCOS_MEDIDA
    assert sorted(blocos) == list(range(1, N_BLOCOS + 1))
    assert H.BLOCOS_MEDIDA_FALHA == tuple(range(1, N_BLOCOS + 1))     # a falha não calibra


def test_falha_acima_do_pico_da_normal_para_em_vez_de_dar_a_volta():
    # a escala vem só da normal: um bloco de falha 5 % acima do pico dela passaria
    # do int16 e, sem a guarda, viraria -31.819
    escala = 0.98 * config.INT16_FULL / 0.05
    normal = _tons([300] * N_BLOCOS).reshape(N_BLOCOS, N_BLOCO)
    H.preparar({"h": normal}, config.FS_TRABALHO, escala)              # no pico: passa
    with pytest.raises(ValueError, match="passam do int16"):
        H.preparar({"bf": 1.05 * normal}, config.FS_TRABALHO, escala)
