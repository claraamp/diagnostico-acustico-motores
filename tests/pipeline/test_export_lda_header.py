"""
Testes do 06_export_lda_header.py (carregado por caminho, porque o nome começa com
dígito). Não dependem de data/: usam modelos sintéticos e o que está versionado em
reports/ (lda_final.json, escores_referencia.csv, lda_modelo.h e reference_data.h).

Garantias: o .h é paramétrico no número de características; lido de volta, traz o
float32 exato do JSON; compilado em C, dá o escore do JSON dentro do limite do
arredondamento em float32 e a mesma regra de empate; compila para o Cortex-M4 sem
promoção a double. O .h versionado é o do lda_final.json atual e reproduz o
segmento 0 do escores_referencia.csv a partir do MFCC do reference_data.h.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "export_lda_header", RAIZ / "scripts" / "pipeline" / "06_export_lda_header.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

ORIGEM = {"json": "modelo.json", "sha256": "0" * 64, "commit": "abc1234"}
MODELO_FINAL = RAIZ / "reports" / "modelo_final"


def _modelo(n: int, seed: int = 0, escore_zero: str = "falha") -> dict:
    rng = np.random.default_rng(seed)
    return {"id": "teste", "versao_formato": 4, "modelo": "lda",
            "escore": "escore >= 0 → falha; escore < 0 → normal",
            "classe_positiva": "falha", "escore_zero": escore_zero,
            "features": [f"f{i}" for i in range(n)],
            "dobrada": {"pesos": (rng.normal(size=n) * 300).tolist(), "bias": float(rng.normal() * 2000)}}


@pytest.mark.parametrize("n", [1, 26, 30])
def test_parametrico_em_n_features(n):
    p = _modelo(n)
    h = H.ler_header(H.gerar_header(p, ORIGEM))
    assert h["n"] == n
    assert np.array_equal(h["pesos"], np.asarray(p["dobrada"]["pesos"], dtype=np.float32))
    assert h["bias"] == np.float32(p["dobrada"]["bias"])


def test_modelo_versionado_lido_de_volta():
    caminho = RAIZ / "reports" / "modelo_final" / "lda_final.json"
    p = json.loads(caminho.read_text(encoding="utf-8"))
    texto = H.gerar_header(p, ORIGEM)
    h = H.ler_header(texto)
    assert h["n"] == len(p["features"])
    assert np.array_equal(h["pesos"], np.asarray(p["dobrada"]["pesos"], dtype=np.float32))
    assert h["empate_e_falha"] == (p["escore_zero"] == p["classe_positiva"])
    assert all(f"{i:2d} {nome} */" in texto for i, nome in enumerate(p["features"]))


@pytest.mark.parametrize("escore_zero, operador", [("falha", ">="), ("normal", ">")])
def test_regra_de_empate_segue_o_json(escore_zero, operador):
    texto = H.gerar_header(_modelo(4, escore_zero=escore_zero), ORIGEM)
    assert f"return escore {operador} 0.0f;" in texto
    h = H.ler_header(texto)
    assert H.prever_header(h, np.array([0.0], dtype=np.float32))[0] == escore_zero


def _sem_ultima_feature(p):
    p["features"] = p["features"][:-1]


def _classe_positiva_normal(p):
    p["classe_positiva"] = "normal"


def _peso_fora_do_float32(p):
    p["dobrada"]["pesos"][0] = 1e39


@pytest.mark.parametrize("estraga", [_sem_ultima_feature, _classe_positiva_normal, _peso_fora_do_float32])
def test_recusa_json_inconsistente(estraga):
    p = _modelo(5)
    estraga(p)
    with pytest.raises(SystemExit):
        H.gerar_header(p, ORIGEM)


def test_header_em_ascii():
    caminho = RAIZ / "reports" / "modelo_final" / "lda_final.json"
    assert H.gerar_header(json.loads(caminho.read_text(encoding="utf-8")), ORIGEM).isascii()
    p = _modelo(2)
    p["features"][0] = "média_c0"
    with pytest.raises(ValueError):
        H.gerar_header(p, ORIGEM)


def test_git_com_erro_nao_conta_como_commitado(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                 # fora de qualquer repositório
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert H.nao_commitados([tmp_path / "modelo.json"])


@pytest.mark.skipif(H.compilador_c() is None, reason="sem compilador C no host")
def test_c_compilado_reproduz_o_json():
    p = _modelo(26, seed=1)
    X = np.random.default_rng(2).normal(size=(200, 26)) * 3
    ref = X @ np.asarray(p["dobrada"]["pesos"]) + p["dobrada"]["bias"]
    texto = H.gerar_header(p, ORIGEM)
    s_c, d_c = H.escore_em_c(texto, X, H.compilador_c())
    assert np.all(np.abs(s_c - ref) <= H.limite_arredondamento(p, X, ref))
    h = H.ler_header(texto)
    assert np.allclose(s_c, H.escore_float32(h, X), rtol=1e-6, atol=1e-3)   # Python imita o C
    assert np.array_equal(d_c == 1, H.escore_float32(h, X) >= 0)


@pytest.mark.skipif(H.compilador_c() is None, reason="sem compilador C no host")
def test_c_empate_exato_e_falha():
    p = _modelo(3)
    p["dobrada"] = {"pesos": [0.0, 0.0, 0.0], "bias": 0.0}       # escore exatamente 0
    _, d_c = H.escore_em_c(H.gerar_header(p, ORIGEM), np.ones((2, 3)), H.compilador_c())
    assert d_c.tolist() == [1, 1]


@pytest.mark.skipif(shutil.which("arm-none-eabi-gcc") is None, reason="sem arm-none-eabi-gcc")
def test_compila_para_cortex_m4(tmp_path):
    (tmp_path / "lda_modelo.h").write_text(H.gerar_header(_modelo(26), ORIGEM), encoding="utf-8")
    (tmp_path / "uso.c").write_text('#include "lda_modelo.h"\n'
                                    "int classificar(const float *x) { return lda_e_falha(lda_escore(x)); }\n",
                                    encoding="utf-8")
    r = subprocess.run(["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16",
                        "-mfloat-abi=hard", *H.CFLAGS, "-c", "-o", str(tmp_path / "uso.o"),
                        str(tmp_path / "uso.c")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_limite_pega_erro_que_1e_3_deixaria_passar():
    """Um bias errado em 1 passa na tolerância relativa de 1e-3, mas não no limite do arredondamento."""
    p = _modelo(26, seed=3)
    X = np.random.default_rng(4).normal(size=(50, 26)) * 3
    ref = X @ np.asarray(p["dobrada"]["pesos"]) + p["dobrada"]["bias"]
    prev = np.where(ref >= 0, "falha", "normal")
    assert H.aprovada(H.conferir(H.gerar_header(p, ORIGEM), p, X, ref, prev, None))
    errado = {**p, "dobrada": {**p["dobrada"], "bias": p["dobrada"]["bias"] + 1.0}}
    texto = H.gerar_header(errado, ORIGEM)
    s = H.escore_float32(H.ler_header(texto), X)
    grandes = np.abs(ref) > 1000                        # onde o 1e-3 relativo aceitaria o erro
    assert grandes.sum() > 10 and np.all(np.abs(s - ref)[grandes] / np.abs(ref[grandes]) < 1e-3)
    with pytest.raises(SystemExit):                     # os valores do .h não são os do JSON
        H.conferir(texto, p, X, ref, prev, None)
    c = H.conferir(texto, errado, X, ref, prev, None)   # .h coerente com um JSON errado
    assert not H.aprovada(c)


def test_lda_modelo_versionado_e_o_do_json_atual():
    """Se o 05 rodar de novo e o 06 não, o .h versionado fica velho: o SHA-256 dele deixa de bater."""
    texto = (MODELO_FINAL / "lda_modelo.h").read_text(encoding="ascii")
    sha = re.search(r"SHA-256 do JSON: ([0-9a-f]{64})", texto).group(1)
    assert sha == hashlib.sha256((MODELO_FINAL / "lda_final.json").read_bytes()).hexdigest(), \
        "lda_modelo.h desatualizado: rode o 06_export_lda_header.py"
    p = json.loads((MODELO_FINAL / "lda_final.json").read_text(encoding="utf-8"))
    h = H.ler_header(texto)
    assert np.array_equal(h["pesos"], np.asarray(p["dobrada"]["pesos"], dtype=np.float32))


@pytest.mark.skipif(H.compilador_c() is None, reason="sem compilador C no host")
def test_lda_modelo_versionado_reproduz_o_segmento_0():
    """
    O critério da tarefa sem data/: o reference_data.h traz o MFCC do segmento 0
    (98 quadros × 13 coeficientes); média e desvio populacional dão as 26 features, e o
    .h versionado, compilado, tem que dar o escore da linha 0 do escores_referencia.csv.
    """
    ref_h = (RAIZ / "reports" / "c_reference" / "reference_data.h").read_text(encoding="ascii")
    quadros = int(re.search(r"#define REF_MFCC_FRAMES (\d+)", ref_h).group(1))
    coefs = int(re.search(r"#define REF_MFCC_COEFS (\d+)", ref_h).group(1))
    corpo = re.search(r"ref_mfcc_matrix\[[^]]*\] = \{(.*?)\};", ref_h, re.S).group(1)
    mfcc = np.array([float(v.strip().rstrip("f")) for v in corpo.split(",") if v.strip()])
    mfcc = mfcc.reshape(quadros, coefs)
    x = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])[None, :]

    linha = next(csv.DictReader((MODELO_FINAL / "escores_referencia.csv").open(encoding="utf-8")))
    assert linha["segmento"] == "0"
    ref = np.array([float(linha["escore"])])
    p = json.loads((MODELO_FINAL / "lda_final.json").read_text(encoding="utf-8"))
    assert x.shape[1] == len(p["features"])

    s_c, d_c = H.escore_em_c((MODELO_FINAL / "lda_modelo.h").read_text(encoding="ascii"), x,
                             H.compilador_c())
    assert np.all(np.abs(s_c - ref) <= H.limite_arredondamento(p, x, ref))
    assert ("falha" if d_c[0] == 1 else "normal") == linha["previsao"]
