#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_export_lda_header.py — gera o .h da LDA final para o firmware
================================================================

Lê o `reports/modelo_final/lda_final.json` do `05` e gera um cabeçalho C com a
forma dobrada do modelo (`escore = pesos · x + bias`, a padronização já
incorporada), em float32, e com a regra de decisão do JSON (**escore ≥ 0 →
falha**, o empate do sklearn). O número de características vem do JSON
(`LDA_N_FEATURES`), sem 26 fixo: o `.h` é regenerado se o modelo final mudar
(pente harmônico, modelo treinado com aumento).

O `.h` não depende da CMSIS: usa `float`, que é o `float32_t` dela, para compilar
também no host. Os valores são escritos com 9 dígitos significativos, que
reproduzem exatamente o float32 mais próximo do valor do JSON.

**Conferência.** Antes de gravar, o script lê o `.h` de volta e confere que os
pesos são o float32 do JSON. Depois calcula o escore de cada segmento com as
features de referência (as do `04`, não versionadas) de dois jeitos: em Python,
com a mesma soma em float32 do C do host, e compilando o `.h` com o compilador C do host,
quando houver um. Os dois têm que dar a mesma previsão do `escores_referencia.csv`
em todos os segmentos e ficar dentro do limite do arredondamento da soma em float32
(`limite_arredondamento`); senão, aborta. Como as features são as mesmas do Python,
o único erro esperado é esse arredondamento, e o limite acompanha `N_FEATURES` e o
cancelamento entre os termos de cada segmento. A tolerância relativa de 1e-3 do
`uso_no_firmware` do JSON é a do firmware, que calcula também o MFCC em float32, e
não serve aqui: deixaria passar um erro de ~1,3 no escore. No Cortex-M4 o gcc funde
multiplicação e soma (`vfma.f32`), e o escore da placa não é bit a bit o do host,
mas fica dentro do mesmo limite. Antes disso, confere que as
features são as do modelo: o escore do JSON em float64 tem que reproduzir o CSV.
Os segmentos são os do próprio treino: isto confere o porte, não mede desempenho.

**Rastreabilidade.** O `.h` traz o id da rodada do modelo, a versão do formato, o
SHA-256 do JSON e o commit em que foi gerado. Numa rodada registrada, o script
aborta se o JSON ou este script tiverem mudanças não commitadas (commit antes de
gerar) e escreve uma linha no registry com a diferença em float32.

Saída: reports/modelo_final/lda_modelo.h (com --sem-registro, em teste/)

Uso
---
    python scripts/pipeline/06_export_lda_header.py --responsavel <nome>
    python scripts/pipeline/06_export_lda_header.py --sem-registro     # teste: grava em teste/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np

import config
import experimentos

U_FLOAT32 = 2.0 ** -24          # arredondamento unitário do float32
REL_CSV = 5e-9                  # o CSV do 05 grava o escore com 9 dígitos (%.9g)
CFLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-Wdouble-promotion"]


def validar_modelo(p: dict) -> int:
    """Confere o que o .h precisa do JSON e devolve o número de características."""
    erros = []
    if p.get("modelo") != "lda":
        erros.append(f"modelo={p.get('modelo')}, esperado lda")
    pesos, nomes = p.get("dobrada", {}).get("pesos", []), p.get("features", [])
    if not pesos or len(pesos) != len(nomes):
        erros.append(f"{len(pesos)} pesos e {len(nomes)} nomes de features")
    valores = np.asarray([*pesos, p.get("dobrada", {}).get("bias", np.nan)], dtype=float)
    with np.errstate(over="ignore"):
        if not np.all(np.isfinite(valores.astype(np.float32))):
            erros.append("pesos ou bias não finitos em float32")
    if p.get("classe_positiva") != "falha":
        erros.append(f"classe_positiva={p.get('classe_positiva')}; o .h supõe escore positivo = falha")
    if p.get("escore_zero") not in ("falha", "normal"):
        erros.append(f"escore_zero={p.get('escore_zero')} não é falha nem normal")
    if erros:
        raise SystemExit("Abortado: JSON do modelo inválido para o .h:\n  " + "\n  ".join(erros))
    return len(pesos)


def _f32(v: float) -> str:
    """Literal C de um float32: 9 dígitos significativos reproduzem o valor exato."""
    s = f"{float(np.float32(v)):.9g}"
    if not any(c in s for c in ".eEn"):
        s += ".0"
    return s + "f"


def gerar_header(p: dict, origem: dict) -> str:
    """Texto do .h a partir do JSON; `origem` traz o caminho, o SHA-256 e o commit."""
    n = validar_modelo(p)
    pesos, nomes = p["dobrada"]["pesos"], p["features"]
    empate_falha = p["escore_zero"] == p["classe_positiva"]
    largura = max(len(_f32(w)) for w in pesos) + 1
    linhas_pesos = [f"    {_f32(w) + ',':<{largura}}  /* {i:2d} {nome} */"
                    for i, (w, nome) in enumerate(zip(pesos, nomes))]
    regra = "escore >= 0 -> falha" if empate_falha else "escore > 0 -> falha"
    texto = "\n".join([
        "/*",
        " * lda_modelo.h -- LDA final do detector binario (normal/falha)",
        " *",
        " * GERADO por scripts/pipeline/06_export_lda_header.py; nao editar a mao.",
        f" * Modelo:  {origem['json']} ({p.get('id', '?')}, versao_formato {p.get('versao_formato', '?')})",
        f" * SHA-256 do JSON: {origem['sha256']}",
        f" * Commit do gerador: {origem['commit']}",
        " *",
        f" * escore = bias + soma(pesos[i] * x[i]), i = 0..LDA_N_FEATURES-1; {regra}.",
        " * A padronizacao ja esta incorporada nos pesos (forma dobrada do JSON). As",
        " * caracteristicas seguem a ordem dos comentarios abaixo, a mesma do JSON.",
        " * O tamanho do escore nao e confianca: usar so o sinal.",
        " */",
        "",
        "#ifndef LDA_MODELO_H",
        "#define LDA_MODELO_H",
        "",
        f"#define LDA_N_FEATURES {n}",
        f"#define LDA_EMPATE_E_FALHA {1 if empate_falha else 0}",
        "",
        "static const float lda_pesos[LDA_N_FEATURES] = {",
        *linhas_pesos,
        "};",
        "",
        f"static const float lda_bias = {_f32(p['dobrada']['bias'])};",
        "",
        "/* Escore da LDA para um vetor de LDA_N_FEATURES caracteristicas. */",
        "static inline float lda_escore(const float x[LDA_N_FEATURES])",
        "{",
        "    float s = lda_bias;",
        "    for (int i = 0; i < LDA_N_FEATURES; ++i) {",
        "        s += lda_pesos[i] * x[i];",
        "    }",
        "    return s;",
        "}",
        "",
        f"/* 1 = falha, 0 = normal ({regra}, como o modelo validado). */",
        "static inline int lda_e_falha(float escore)",
        "{",
        f"    return escore {'>=' if empate_falha else '>'} 0.0f;",
        "}",
        "",
        "#endif /* LDA_MODELO_H */",
        "",
    ])
    if not texto.isascii():      # como o reference_data.h: compiladores embarcados e UTF-8
        raise ValueError("o .h tem caracteres fora do ASCII (nomes de features no JSON?)")
    return texto


def ler_header(texto: str) -> dict:
    """Lê de volta do .h o número de características, os pesos, o bias e a regra."""
    n = int(re.search(r"#define LDA_N_FEATURES (\d+)", texto).group(1))
    corpo = re.search(r"lda_pesos\[LDA_N_FEATURES\] = \{(.*?)\};", texto, re.S).group(1)
    corpo = re.sub(r"/\*.*?\*/", "", corpo, flags=re.S)
    pesos = np.array([np.float32(v.strip().rstrip("f")) for v in corpo.split(",") if v.strip()],
                     dtype=np.float32)
    bias = np.float32(re.search(r"lda_bias = ([^;]+?)f?;", texto).group(1))
    empate = int(re.search(r"#define LDA_EMPATE_E_FALHA (\d)", texto).group(1))
    if len(pesos) != n:
        raise ValueError(f"o .h declara {n} características e traz {len(pesos)} pesos")
    return {"n": n, "pesos": pesos, "bias": bias, "empate_e_falha": bool(empate)}


def escore_float32(h: dict, X: np.ndarray) -> np.ndarray:
    """
    A soma do `lda_escore` em float32, na mesma ordem, arredondando o produto e a soma
    separadamente, como o C do host (x86-64 sem FMA). No Cortex-M4 o gcc funde os dois
    num `vfma.f32`, então o escore da placa difere deste no último bit de cada passo;
    a diferença fica muito abaixo da tolerância, mas não é bit a bit.
    """
    X32 = np.asarray(X, dtype=np.float32)
    s = np.full(len(X32), h["bias"], dtype=np.float32)
    for i in range(h["n"]):
        s = (s + h["pesos"][i] * X32[:, i]).astype(np.float32)
    return s


def limite_arredondamento(p: dict, X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Limite, por segmento, da diferença entre o escore em float32 e o do CSV quando as
    features são as mesmas: arredondar x, os pesos e o bias para float32 e somar os
    N produtos em sequência erra no máximo (N + 3)·u·(Σ|pesos·x| + |bias|), e o CSV
    acrescenta o arredondamento dos seus 9 dígitos.
    """
    pesos = np.asarray(p["dobrada"]["pesos"], dtype=float)
    termos = np.abs(np.asarray(X, dtype=float) * pesos).sum(axis=1) + abs(p["dobrada"]["bias"])
    return (len(pesos) + 3) * U_FLOAT32 * termos + REL_CSV * np.abs(ref)


def prever_header(h: dict, escores: np.ndarray) -> np.ndarray:
    falha = escores >= 0 if h["empate_e_falha"] else escores > 0
    return np.where(falha, "falha", "normal")


def compilador_c() -> str | None:
    return next((c for c in ("cc", "gcc", "clang") if shutil.which(c)), None)


def versao_compilador(cc: str | None) -> str:
    """Primeira linha do `--version`, sem os separadores do registry."""
    if cc is None:
        return "nenhum"
    r = subprocess.run([cc, "--version"], capture_output=True, text=True)
    linha = (r.stdout.splitlines() or [cc])[0]
    return re.sub(r"[;,=]", " ", linha).strip()


_PROGRAMA_C = """\
#include <stdio.h>
#include "lda_modelo.h"
int main(void)
{
    float x[LDA_N_FEATURES];
    while (fread(x, sizeof(float), LDA_N_FEATURES, stdin) == LDA_N_FEATURES) {
        float s = lda_escore(x);
        printf("%.9g %d\\n", (double)s, lda_e_falha(s));
    }
    return 0;
}
"""


def escore_em_c(texto: str, X: np.ndarray, cc: str) -> tuple[np.ndarray, np.ndarray]:
    """Compila o .h no host e devolve o escore e a decisão (1 = falha) de cada linha de X."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "lda_modelo.h").write_text(texto, encoding="utf-8")
        (tmp / "main.c").write_text(_PROGRAMA_C, encoding="utf-8")
        r = subprocess.run([cc, *CFLAGS, "-o", str(tmp / "lda"), str(tmp / "main.c")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"Abortado: o .h não compila com {cc}:\n{r.stderr}")
        r = subprocess.run([str(tmp / "lda")], input=np.asarray(X, dtype="<f4").tobytes(),
                           check=True, capture_output=True)
    saida = np.array([l.split() for l in r.stdout.decode().splitlines()], dtype=float)
    return saida[:, 0], saida[:, 1].astype(int)


def carregar_referencia(features: Path, escores: Path, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X do 04 e escore/previsão do CSV do 05, conferidos contra o JSON."""
    manifesto_path = features.with_name("manifest_features.json")
    for caminho in (features, manifesto_path, escores):
        if not caminho.exists():
            raise SystemExit(f"Abortado: {caminho} não encontrado (a conferência precisa das "
                             "features do 04 e dos escores do 05).")
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    if manifesto.get("splits_hash") != p["treino"]["splits"] or manifesto.get("norm_clipe"):
        raise SystemExit("Abortado: as features não são as de referência do modelo "
                         f"(splits {manifesto.get('splits_hash')} × {p['treino']['splits']}, "
                         f"norm_clipe={manifesto.get('norm_clipe')}).")
    X = np.load(features)["X"]
    linhas = list(csv.DictReader(escores.open(encoding="utf-8")))
    ref = np.array([float(l["escore"]) for l in linhas])
    prev = np.array([l["previsao"] for l in linhas])
    if X.shape != (len(ref), len(p["features"])):
        raise SystemExit(f"Abortado: X tem forma {X.shape}; o CSV tem {len(ref)} segmentos e o "
                         f"modelo, {len(p['features'])} características.")
    s64 = X @ np.asarray(p["dobrada"]["pesos"]) + p["dobrada"]["bias"]
    if np.max(np.abs(s64 - ref) / np.maximum(np.abs(ref), 1.0)) > 1e-6:
        raise SystemExit("Abortado: o JSON em float64 não reproduz o escores_referencia.csv com "
                         "estas features; rode o 04 sem opções.")
    return X, ref, prev


def conferir(texto: str, p: dict, X: np.ndarray, ref: np.ndarray, prev: np.ndarray,
             cc: str | None) -> dict:
    """Confere o .h lido de volta contra o JSON e o escore em float32 (Python e C) contra o CSV."""
    h = ler_header(texto)
    pesos32 = np.asarray(p["dobrada"]["pesos"], dtype=np.float32)
    if h["n"] != len(pesos32) or not np.array_equal(h["pesos"], pesos32) \
            or h["bias"] != np.float32(p["dobrada"]["bias"]):
        raise SystemExit("Abortado: os valores lidos do .h não são o float32 do JSON.")
    escala = np.maximum(np.abs(ref), 1.0)
    limite = limite_arredondamento(p, X, ref)
    s_py = escore_float32(h, X)
    out = {
        "n_segmentos": len(ref),
        "min_abs_escore": float(np.min(np.abs(ref))),
        "max_dif_escore_f32": float(np.max(np.abs(s_py - ref))),
        "max_dif_rel_f32": float(np.max(np.abs(s_py - ref) / escala)),
        "max_dif_sobre_limite_f32": float(np.max(np.abs(s_py - ref) / limite)),
        "previsoes_iguais_f32": bool(np.array_equal(prever_header(h, s_py), prev)),
        "compilador_c": versao_compilador(cc),
    }
    if cc:
        s_c, d_c = escore_em_c(texto, X, cc)
        out["max_dif_escore_c"] = float(np.max(np.abs(s_c - ref)))
        out["max_dif_rel_c"] = float(np.max(np.abs(s_c - ref) / escala))
        out["max_dif_sobre_limite_c"] = float(np.max(np.abs(s_c - ref) / limite))
        out["previsoes_iguais_c"] = bool(np.array_equal(np.where(d_c == 1, "falha", "normal"), prev))
    return out


def aprovada(c: dict) -> bool:
    """Mesma previsão em todos os segmentos e diferença dentro do limite do arredondamento."""
    ok = c["previsoes_iguais_f32"] and c["max_dif_sobre_limite_f32"] <= 1.0
    if "max_dif_sobre_limite_c" in c:
        ok = ok and c["previsoes_iguais_c"] and c["max_dif_sobre_limite_c"] <= 1.0
    return ok


def nao_commitados(caminhos: list[Path]) -> list[str]:
    """
    Caminhos com mudanças em relação ao HEAD. Se o git falhar (fora do repositório,
    git ausente), não dá para garantir o commit: devolve o erro como pendência.
    """
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", *map(str, caminhos)],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return [f"(git indisponível: {e})"]
    if r.returncode != 0:
        return [f"(git status falhou: {r.stderr.strip()})"]
    return [l[3:] for l in r.stdout.splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", type=Path, default=Path("reports/modelo_final/lda_final.json"))
    ap.add_argument("--escores", type=Path, default=Path("reports/modelo_final/escores_referencia.csv"))
    ap.add_argument("--features", type=Path,
                    default=Path(config.dir_features(config.FS_TRABALHO)) / "mfcc_features.npz")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/modelo_final"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    ap.add_argument("--sem-registro", action="store_true",
                    help="teste: grava em <out-dir>/teste/ e não escreve no registry")
    args = ap.parse_args()
    if not args.sem_registro and not args.responsavel:
        ap.error("--responsavel é obrigatório numa rodada registrada")

    if not args.sem_registro:
        sujos = nao_commitados([args.modelo, Path(__file__)])
        if sujos:
            print("Abortado: commit antes de gerar; há mudanças não commitadas em:\n  "
                  + "\n  ".join(sujos))
            return 1

    p = json.loads(args.modelo.read_text(encoding="utf-8"))
    n = validar_modelo(p)
    origem = {"json": args.modelo.as_posix(),
              "sha256": hashlib.sha256(args.modelo.read_bytes()).hexdigest(),
              "commit": experimentos.git_short_hash()}
    texto = gerar_header(p, origem)
    print(f"modelo {p.get('id')} (versao_formato {p.get('versao_formato')}): {n} características, "
          f"{p['escore']}")

    X, ref, prev = carregar_referencia(args.features, args.escores, p)
    cc = compilador_c()
    if cc is None:
        print("aviso: nenhum compilador C no host; a conferência fica só com a soma em float32 do Python")
    c = conferir(texto, p, X, ref, prev, cc)
    print("conferência com o escores_referencia.csv: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    if not aprovada(c):
        print("Abortado: o .h não reproduz a referência (previsões idênticas e diferença dentro do "
              "limite do arredondamento em float32).")
        return 1

    destino = args.out_dir / "teste" if args.sem_registro else args.out_dir
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "lda_modelo.h").write_text(texto, encoding="utf-8")
    print(f"cabeçalho: {destino}/lda_modelo.h")

    if args.sem_registro:
        print("(rodada de teste: registry não alterado)")
        return 0
    exp_id = f"exp{experimentos.next_exp_number(args.registry):03d}"
    experimentos.append_registry(args.registry, [{
        "id": exp_id,
        "data": date.today().isoformat(),
        "etapa": "exportacao_firmware",
        "script": "pipeline/06_export_lda_header.py",
        "git_commit": origem["commit"],
        "parametros": experimentos.kv({
            "modelo": p.get("id"), "versao_formato": p.get("versao_formato"),
            "n_features": n, "tipo": "float32",
            "empate_e_falha": p["escore_zero"] == p["classe_positiva"],
            "json_sha256": origem["sha256"][:12], "compilador_c": c["compilador_c"],
        }),
        "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
        "metricas": experimentos.kv({
            "n_segmentos": c["n_segmentos"],
            "max_dif_escore_f32": f"{c['max_dif_escore_f32']:.1e}",
            "max_dif_escore_c": f"{c['max_dif_escore_c']:.1e}" if "max_dif_escore_c" in c else None,
            "max_dif_sobre_limite": f"{max(c['max_dif_sobre_limite_f32'], c.get('max_dif_sobre_limite_c', 0.0)):.2f}",
            "min_abs_escore": f"{c['min_abs_escore']:.0f}",
            "previsoes_iguais": c["previsoes_iguais_f32"] and c.get("previsoes_iguais_c", True),
        }),
        "responsavel": args.responsavel,
        "notas": args.notas or ("cabeçalho C da LDA final (forma dobrada, float32), conferido contra o "
                                "escores_referencia.csv; segmentos do treino, não é desempenho"),
    }])
    print(f"registrado: {exp_id} em {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
