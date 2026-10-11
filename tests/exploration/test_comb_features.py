"""
Testes do pente harmônico (inspect_comb_features.py), com sinais sintéticos.

Garante que (1) nos candidatos viáveis, o pente acha o f₀ de uma série fora da
razão cinemática, dentro da busca de ±3 %, e pontua alto; (2) sem série, a
pontuação fica baixa; (3) a estimativa do eixo não cai em BPFO/4 quando a série da
BPFO é forte; (4) o f₀ não entra nas features; (5) o candidato A (espectro médio dos
quadros) é o espectro da cadeia do MFCC e é inviável pela máscara do eixo; e (6) a
decisão segue o critério do card: o pente só entra se ganhar por mais que a margem
em todas as falhas externas, sem perder especificidade.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_CAMINHO = Path(__file__).resolve().parents[2] / "scripts" / "exploration" / "inspect_comb_features.py"
_spec = importlib.util.spec_from_file_location("inspect_comb_features", _CAMINHO)
P = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P          # o @dataclass do módulo procura o próprio módulo aqui
_spec.loader.exec_module(P)

FS = 12_800
EIXO = 50.20
VIAVEIS = [c for c in P.CANDIDATOS if P.inviavel(c) is None]


def _sinal(bpfo: float | None, amp: float = 0.01, eixo_amp: float = 0.003, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(FS) / FS
    x = 0.01 * rng.standard_normal(FS)
    for k in range(1, 40):
        x += eixo_amp / k * np.sin(2 * np.pi * k * EIXO * t + rng.uniform(0, 6.28))
    if bpfo:
        for k in range(1, int(5600 // bpfo) + 1):
            x += amp * np.sin(2 * np.pi * k * bpfo * t + rng.uniform(0, 6.28))
    return x


def _pente_bpfo(x, c):
    f, p = P.espectro(x, c)
    pico, fundo = P.pico_e_fundo(f, p, c.fundo_hz)
    return P.pente(f, pico, fundo, EIXO, P.RAZAO_BPFO, c)


def test_candidatos_do_card():
    nomes = [c.nome[0] for c in P.CANDIDATOS]
    assert nomes == ["A", "B", "C"]
    a, b, c = P.CANDIDATOS
    assert (a.n_bloco, a.n_fft, a.passo, a.n_blocos) == (320, 512, 128, 98)
    assert (b.n_bloco, b.n_fft, b.n_blocos) == (4096, 4096, 3) and b.resolucao == pytest.approx(3.125)
    assert c.n_bloco == FS and c.resolucao == pytest.approx(1.0)


def test_candidato_a_e_o_espectro_da_cadeia_e_inviavel():
    a = P.CANDIDATOS[0]
    x = _sinal(182.65)
    quadros = np.array([x[128 * k: 128 * k + 320] * np.hanning(320) for k in range(98)])
    esperado = (np.abs(np.fft.rfft(quadros, n=512, axis=1)) ** 2).mean(axis=0)
    np.testing.assert_allclose(P.espectro(x, a)[1], esperado, rtol=1e-12)
    assert P.inviavel(a) is not None
    with pytest.raises(ValueError):          # nenhum dente fica fora da máscara do eixo
        _pente_bpfo(x, a)


@pytest.mark.parametrize("c", VIAVEIS, ids=lambda c: c.nome)
def test_acha_serie_fora_da_razao_cinematica(c):
    r = _pente_bpfo(_sinal(182.65), c)          # +1,8 % da cinemática
    assert abs(r["f0"] - 182.65) < 0.5
    assert r["pontuacao"] > 10 and r["fracao"] > 0.8


@pytest.mark.parametrize("c", VIAVEIS, ids=lambda c: c.nome)
def test_sem_serie_pontua_baixo(c):
    com, sem = _pente_bpfo(_sinal(182.65), c), _pente_bpfo(_sinal(None), c)
    assert sem["pontuacao"] < 8
    assert com["pontuacao"] - sem["pontuacao"] > 8


@pytest.mark.parametrize("c", VIAVEIS, ids=lambda c: c.nome)
def test_eixo_nao_cai_em_bpfo_sobre_4(c):
    # Com a série da BPFO forte, um pente do eixo com muitos múltiplos se alinha em
    # BPFO/4 (≈ 45,7 Hz). A estimativa em duas etapas não pode cair lá.
    f, p = P.espectro(_sinal(182.65, amp=0.05), c)
    pico, fundo = P.pico_e_fundo(f, p, c.fundo_hz)
    est = P.estimar_eixo(f, pico, fundo)
    assert abs(est - 182.65 / 4) > 2.0
    assert abs(est - EIXO) < 2.5


def test_f0_nao_e_feature():
    v, diag = P.features_segmento(_sinal(182.65), VIAVEIS[0], "fixo")
    assert v.shape == (len(P.NOMES_FEATURES),) == (4,)
    assert "f0_bpfo" in diag
    assert not np.isclose(v, diag["f0_bpfo"]).any()
    with pytest.raises(ValueError):
        P.features_segmento(_sinal(182.65), VIAVEIS[0], "outro")


def _linhas(medias: dict, espec: dict | None = None) -> list[dict]:
    """Linhas de validação interna sintéticas: uma por (falha externa, conjunto)."""
    espec = espec or {}
    return [{"externo": e, "falha_interna": "x", "conjunto": c, "sensibilidade": m,
             "especificidade": espec.get((e, c), 1.0), "acuracia_balanceada": m}
            for (e, c), m in medias.items()]


EXTERNAS = ("bpfi_0.3mm", "bpfi_1.0mm", "bpfo_0.3mm", "bpfo_1.0mm")


def test_decisao_pente_entra_se_ganhar_alem_da_margem_em_todas():
    m = {(e, c): (0.90 if c == "mfcc" else 0.95) for e in EXTERNAS for c in ("mfcc", "mfcc+pente")}
    assert P.decidir(_linhas(m))["entra"]


def test_decisao_dentro_da_margem_fica_o_mfcc():
    m = {(e, c): (0.90 if c == "mfcc" else 0.915) for e in EXTERNAS for c in ("mfcc", "mfcc+pente")}
    d = P.decidir(_linhas(m))
    assert not d["entra"] and "nenhuma" in d["situacao"]


def test_decisao_dividida_fica_fora():
    m = {(e, c): (0.95 if c == "mfcc+pente" and e == "bpfo_0.3mm" else 0.90)
         for e in EXTERNAS for c in ("mfcc", "mfcc+pente")}
    d = P.decidir(_linhas(m))
    assert not d["entra"] and "dividido" in d["situacao"]


def test_decisao_nao_aceita_perder_especificidade():
    m = {(e, c): (0.90 if c == "mfcc" else 0.95) for e in EXTERNAS for c in ("mfcc", "mfcc+pente")}
    esp = {(e, "mfcc+pente"): 0.98 for e in EXTERNAS}
    d = P.decidir(_linhas(m, esp))
    assert not d["entra"]
    assert all(x["perde_especificidade"] for x in d["por_externo"].values())


def test_custo_b_cabe_na_cmsis_e_c_nao():
    _, b, c = P.CANDIDATOS
    assert P.custo_stm32(b)["fft_na_cmsis"] and not P.custo_stm32(c)["fft_na_cmsis"]
    assert P.custo_stm32(b)["ram_kb"] == pytest.approx(4 * (2 * 4096 + 2049) / 1024)
