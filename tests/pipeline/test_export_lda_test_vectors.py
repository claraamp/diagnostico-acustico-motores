"""
Testes do 07_export_lda_test_vectors.py (carregado por caminho, porque o nome começa
com dígito). Não dependem de data/: as features são sintéticas, montadas a partir do
segmento 0 do reference_data.h e deslocadas ao longo dos pesos para que o JSON em
float64 reproduza cada linha do escores_referencia.csv versionado.

Garantias: a seleção não olha o escore e inclui o primeiro e o último segmento de cada
gravação; a demonstração tem um segmento por gravação; o .h gerado, lido de volta e
somado em float32 (e compilado em C, se houver compilador), fica dentro da tolerância
e dá a decisão do CSV; o gerador recusa um lda_modelo.h de outro modelo.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / "scripts" / "pipeline" / "07_export_lda_test_vectors.py"
_spec = importlib.util.spec_from_file_location("export_lda_test_vectors", SCRIPT)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

MODELO_FINAL = RAIZ / "reports" / "modelo_final"


def _features_sinteticas(destino: Path) -> Path:
    """X (n_segmentos × 26) cujo escore em float64 é o do CSV; grava npz e manifesto."""
    ref_h = (RAIZ / "reports" / "c_reference" / "reference_data.h").read_text(encoding="ascii")
    q = int(re.search(r"#define REF_MFCC_FRAMES (\d+)", ref_h).group(1))
    c = int(re.search(r"#define REF_MFCC_COEFS (\d+)", ref_h).group(1))
    corpo = re.search(r"ref_mfcc_matrix\[[^]]*\] = \{(.*?)\};", ref_h, re.S).group(1)
    mfcc = np.array([float(v.strip().rstrip("f")) for v in corpo.split(",") if v.strip()]).reshape(q, c)
    x0 = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])

    p = json.loads((MODELO_FINAL / "lda_final.json").read_text(encoding="utf-8"))
    w, b = np.asarray(p["dobrada"]["pesos"]), p["dobrada"]["bias"]
    linhas = list(csv.DictReader((MODELO_FINAL / "escores_referencia.csv").open(encoding="utf-8")))
    ref = np.array([float(l["escore"]) for l in linhas])
    X = x0[None, :] + ((ref - (x0 @ w + b)) / (w @ w))[:, None] * w[None, :]

    destino.mkdir(parents=True, exist_ok=True)
    np.savez(destino / "mfcc_features.npz", X=X)
    (destino / "manifest_features.json").write_text(json.dumps(
        {"splits_hash": p["treino"]["splits"], "norm_clipe": False, "git_commit": "teste00"}),
        encoding="utf-8")
    return destino / "mfcc_features.npz"


def _rodar(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    feats = _features_sinteticas(tmp_path / "features")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--features", str(feats),
         "--out", str(tmp_path / "out" / V.NOME_H), *extra],
        cwd=RAIZ, capture_output=True, text=True)


def test_selecao_nao_olha_o_escore_e_pega_as_pontas():
    g = np.array(["a"] * 10 + ["b"] * 7)
    sel = V.selecionar(g, 3)
    assert list(sel) == [0, 4, 9, 10, 13, 16]   # primeiro, meio e último de cada gravação
    assert list(V.selecionar(g, 0)) == list(range(17))
    assert list(V.selecionar(g, 50)) == list(range(17))


def test_demo_tem_o_menor_escore_de_cada_gravacao():
    g = np.array(["a", "a", "a", "b", "b"])
    ref = np.array([-900.0, -100.0, -500.0, 300.0, 2000.0])
    assert V.demo(g, ref) == [1, 3]


def test_gera_todos_os_segmentos_dentro_da_tolerancia(tmp_path):
    r = _rodar(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    texto = (tmp_path / "out" / V.NOME_H).read_text(encoding="ascii")
    n_csv = sum(1 for _ in (MODELO_FINAL / "escores_referencia.csv").open()) - 1
    assert f"#define VT_N_SEGMENTOS {n_csv}" in texto
    assert "#define VT_N_FEATURES 26" in texto
    assert "#define VT_N_DEMO 5" in texto
    X, ref, tol = V.ler_vetores(texto)
    assert X.shape == (n_csv, 26) and np.all(tol > 0)


def test_por_gravacao_reduz_o_arquivo(tmp_path):
    r = _rodar(tmp_path, "--por-gravacao", "4")
    assert r.returncode == 0, r.stdout + r.stderr
    texto = (tmp_path / "out" / V.NOME_H).read_text(encoding="ascii")
    assert "#define VT_N_SEGMENTOS 20" in texto


def test_recusa_lda_h_de_outro_modelo(tmp_path):
    original = (MODELO_FINAL / "lda_modelo.h").read_text(encoding="ascii")
    outro = re.sub(r"lda_bias = [^;]+;", "lda_bias = 1.0f;", original)
    (tmp_path / "lda_modelo.h").write_text(outro, encoding="ascii")
    r = _rodar(tmp_path, "--lda-h", str(tmp_path / "lda_modelo.h"))
    assert r.returncode != 0
    assert "não é o deste lda_final.json" in r.stdout + r.stderr
    assert not (tmp_path / "out" / V.NOME_H).exists()


def test_tolerancia_cobre_o_arredondamento_do_escore_de_referencia():
    p = json.loads((MODELO_FINAL / "lda_final.json").read_text(encoding="utf-8"))
    X = np.ones((1, len(p["features"])))
    ref = np.array([1500.0])
    assert V.tolerancia(p, X, ref)[0] > V.H.limite_arredondamento(p, X, ref)[0]


@pytest.mark.skipif(V.H.compilador_c() is None, reason="sem compilador C no host")
def test_cabecalhos_compilam_juntos_sem_promocao_a_double(tmp_path):
    r = _rodar(tmp_path, "--por-gravacao", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    (tmp_path / "out" / "lda_modelo.h").write_text(
        (MODELO_FINAL / "lda_modelo.h").read_text(encoding="ascii"), encoding="ascii")
    (tmp_path / "out" / "main.c").write_text(
        '#include "lda_modelo.h"\n#include "lda_vetores_teste.h"\n'
        "_Static_assert(VT_N_FEATURES == LDA_N_FEATURES, \"n\");\n"
        "int main(void){int ok=1;for(int i=0;i<VT_N_SEGMENTOS;i++){float s=lda_escore(vt_x[i]);"
        "float d=s-vt_escore_ref[i];if(d<0.0f)d=-d;ok&=(d<=vt_tolerancia[i]);"
        "ok&=(lda_e_falha(s)==vt_falha_ref[i]);}ok&=(lda_bias==VT_LDA_BIAS);"
        "(void)vt_segmento;(void)vt_gravacao;(void)vt_demo;(void)vt_nomes_gravacao;return ok?0:1;}\n",
        encoding="ascii")
    cc = V.H.compilador_c()
    exe = tmp_path / "out" / "vt"
    c = subprocess.run([cc, *V.H.CFLAGS, "-o", str(exe), str(tmp_path / "out" / "main.c")],
                       capture_output=True, text=True)
    assert c.returncode == 0, c.stderr
    assert subprocess.run([str(exe)]).returncode == 0


def test_commit_antes_de_gerar_so_aborta_no_caminho_versionado(tmp_path):
    abortar, msg = V.checar_commit(V.OUT_VERSIONADO, ["scripts/pipeline/07_export_lda_test_vectors.py"])
    assert abortar and "commit antes de gerar" in msg
    abortar, msg = V.checar_commit(tmp_path / V.NOME_H, ["reports/modelo_final/lda_final.json"])
    assert not abortar and "+mudancas" in msg
    assert V.checar_commit(V.OUT_VERSIONADO, []) == (False, "")
