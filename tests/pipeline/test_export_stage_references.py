"""
Testes do 08_export_stage_references.py (carregado por caminho, porque o nome começa
com dígito). Não dependem de data/: os segmentos são sintéticos, e as referências
versionadas (se já existirem) são conferidas a partir do próprio int16 gravado nelas.

Garantias: o formato volta igual do arquivo (o que o harness em C lê); a seleção
padrão é o segmento do meio de cada gravação, sem olhar o sinal; o tolerancias.txt
é o dsp.ESTAGIOS_PORTE; e as referências versionadas são as do dsp atual, com o
tolerancias.txt atual e os SHA-256 do manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import dsp

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / "scripts" / "pipeline" / "08_export_stage_references.py"
_spec = importlib.util.spec_from_file_location("export_stage_references", SCRIPT)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

VERSIONADO = RAIZ / R.OUT_VERSIONADO


@dataclass
class _Seg:
    id: int
    rotulo: str


def _pcm(semente=7):
    return np.random.default_rng(semente).integers(-12000, 12000, 12800).astype(np.int16)


def test_formato_volta_igual_do_arquivo(tmp_path):
    e = dsp.estagios_mfcc(_pcm())
    e["escore"] = np.array([-1234.5678])
    shas = R.gravar_segmento(tmp_path / "s", e, ["teste"])
    lido = R.ler_segmento(tmp_path / "s")
    assert set(lido) == {n for n, _ in R.ARQUIVOS}
    for nome, _ in R.ARQUIVOS:
        np.testing.assert_array_equal(lido[nome].ravel(), np.asarray(e[nome]).ravel())
    assert lido["janela"].shape == (98, 320) and lido["entrada"].dtype == np.int16
    # binário sem cabeçalho: 8 bytes por valor (2 no int16), little-endian
    assert (tmp_path / "s" / "espectro.f64").stat().st_size == 98 * 257 * 8
    assert (tmp_path / "s" / "entrada.i16").read_bytes()[:2] == int(e["entrada"][0]).to_bytes(2, "little", signed=True)
    assert shas["mfcc.f64"] == hashlib.sha256((tmp_path / "s" / "mfcc.f64").read_bytes()).hexdigest()
    assert b"\r" not in (tmp_path / "s" / "indice.txt").read_bytes()


def test_selecao_padrao_e_o_meio_de_cada_gravacao():
    segs = [_Seg(i, "a") for i in range(5)] + [_Seg(5 + i, "b") for i in range(4)]
    assert [s.id for s in R.selecionar(segs, None)] == [2, 7]
    assert [s.id for s in R.selecionar(segs, [8, 0])] == [8, 0]
    with pytest.raises(SystemExit):
        R.selecionar(segs, [99])


def test_tolerancias_txt_e_o_dsp():
    linhas = [l.split() for l in R.texto_tolerancias().splitlines() if not l.startswith("#")]
    assert [(n, m, float(i), float(c)) for n, m, i, c in linhas] == list(dsp.ESTAGIOS_PORTE)


@pytest.mark.skipif(not (VERSIONADO / "manifest.json").exists(),
                    reason="referências por estágio ainda não geradas")
def test_referencias_versionadas_sao_as_do_dsp_atual():
    """Se o dsp ou as tolerâncias mudarem sem rodar o 08 de novo, este teste falha."""
    manifesto = json.loads((VERSIONADO / "manifest.json").read_text(encoding="utf-8"))
    for relativo, sha in manifesto["arquivos_sha256"].items():
        assert hashlib.sha256((VERSIONADO / relativo).read_bytes()).hexdigest() == sha, relativo
    assert (VERSIONADO / "tolerancias.txt").read_text(encoding="utf-8") == R.texto_tolerancias(), \
        "tolerancias.txt desatualizado: rode o 08_export_stage_references.py"
    pastas = (VERSIONADO / "segmentos.txt").read_text(encoding="utf-8").split()
    assert pastas == manifesto["segmentos"] and len(pastas) == 5
    for pasta in pastas:
        lido = R.ler_segmento(VERSIONADO / pasta)
        e = dsp.estagios_mfcc(lido["entrada"].ravel())
        for nome, _ in R.ARQUIVOS:
            if nome != "escore":
                np.testing.assert_array_equal(lido[nome].ravel(), np.asarray(e[nome]).ravel(),
                                              err_msg=f"{pasta}/{nome}: rode o 08 de novo")
