#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_export_lda_test_vectors.py — vetores de teste da LDA para a placa (marco de 13/10)
=====================================================================================

Gera um cabeçalho C com as características de referência de segmentos do
`04` (as 26 entradas da LDA, em float32), o escore e a decisão do
`escores_referencia.csv` e, por segmento, a diferença máxima admitida entre o
escore da placa e o de referência. O firmware (`lab3-firmware-stm32`) inclui
este `.h` ao lado do `lda_modelo.h`, calcula o escore de cada segmento com o
`lda_escore()`, compara com a referência, acende o LED da decisão e manda o
resultado pela serial.

O marco confere o **porte do classificador**, não o desempenho: as
características vêm do Python (na época do marco de 13/10, o MFCC ainda não
estava portado) e os segmentos são os do próprio treino do modelo final.

**Tolerância.** É a mesma do `06`: `limite_arredondamento()`, o limite do erro
de arredondar x, os pesos e o bias para float32 e somar os N produtos em
sequência, mais o arredondamento dos 9 dígitos do CSV. Aqui soma-se mais um
`u·|ref|`, porque o escore de referência também vai para o `.h` em float32. No
Cortex-M4, o gcc funde multiplicação e soma (`vfma.f32`); isso arredonda menos
vezes e fica dentro do mesmo limite. A tolerância relativa de 1e-3 do
`uso_no_firmware` do JSON é a da cadeia inteira (com o MFCC em float32) e não
serve aqui: o MFCC vem pronto do Python.

**Conferências antes de gravar** (aborta se alguma falhar):

- as características são as do modelo: o JSON em float64 reproduz o CSV
  (`carregar_referencia()` do `06`);
- o `lda_modelo.h` versionado traz o float32 exato dos pesos e do bias do JSON,
  ou seja, é o `.h` deste modelo;
- a soma em float32 do Python, na ordem do `lda_escore()`, fica dentro da
  tolerância em todos os segmentos exportados e dá a mesma decisão do CSV;
- se houver compilador C no host, o mesmo vale para o `lda_modelo.h` compilado
  e alimentado com os valores lidos de volta do `.h` gerado.

**Demonstração com os LEDs.** O `.h` também lista, por gravação, o segmento de
menor |escore| (o mais perto da fronteira). O firmware percorre essa lista em
laço, com uma pausa por segmento, para a demonstração.

**Rastreabilidade.** O `.h` traz o id do modelo, o SHA-256 do JSON e do CSV, o
commit e o bias do modelo (`VT_LDA_BIAS`), que o firmware compara com o
`lda_bias` do `lda_modelo.h` para recusar uma combinação de cabeçalhos de
modelos diferentes. Não há linha no registry: o arquivo é uma fixture de
teste, sem métrica; o resultado do marco é o log da placa.

**Commit antes de gerar.** Como no `06`, o commit escrito no `.h` só identifica
o que o gerou se o script e as entradas versionadas (`lda_final.json`,
`escores_referencia.csv`, `lda_modelo.h`) não tiverem mudanças não commitadas.
Gravando no caminho versionado, o script aborta se houver; com `--out` em outro
lugar (teste), só avisa e marca o commit no `.h` com `+mudancas`.

Saída: reports/modelo_final/lda_vetores_teste.h (--out para outro caminho)

Uso
---
    python scripts/pipeline/07_export_lda_test_vectors.py                  # todos os segmentos
    python scripts/pipeline/07_export_lda_test_vectors.py --por-gravacao 6 # 6 por gravação
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np

import config
import experimentos

_spec = importlib.util.spec_from_file_location(
    "export_lda_header", Path(__file__).resolve().with_name("06_export_lda_header.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

U_FLOAT32 = H.U_FLOAT32
NOME_H = "lda_vetores_teste.h"
OUT_VERSIONADO = Path("reports/modelo_final") / NOME_H


def selecionar(gravacoes: np.ndarray, por_gravacao: int) -> np.ndarray:
    """
    Índices exportados, na ordem do CSV. Com `por_gravacao` = 0, todos; senão,
    `por_gravacao` segmentos igualmente espaçados dentro de cada gravação (o
    primeiro e o último incluídos). Não olha o escore: a escolha não depende do
    resultado.
    """
    if por_gravacao < 0:
        raise ValueError("--por-gravacao não pode ser negativo")
    if por_gravacao == 0:
        return np.arange(len(gravacoes))
    escolhidos = []
    for g in dict.fromkeys(gravacoes):                 # ordem de aparição
        idx = np.flatnonzero(gravacoes == g)
        k = min(por_gravacao, len(idx))
        pos = np.unique(np.round(np.linspace(0, len(idx) - 1, k)).astype(int))
        escolhidos.extend(idx[pos])
    return np.array(sorted(escolhidos))


def demo(gravacoes: np.ndarray, ref: np.ndarray) -> list[int]:
    """Por gravação, a posição (dentro dos exportados) do segmento de menor |escore|."""
    return [int(np.flatnonzero(gravacoes == g)[np.argmin(np.abs(ref[gravacoes == g]))])
            for g in dict.fromkeys(gravacoes)]


def tolerancia(p: dict, X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Limite do 06 mais o arredondamento do escore de referência para float32."""
    return H.limite_arredondamento(p, X, ref) + U_FLOAT32 * np.abs(ref)


def conferir_lda_h(texto_h: str, p: dict) -> dict:
    """O lda_modelo.h tem que ser o float32 exato do JSON (é o .h deste modelo)."""
    h = H.ler_header(texto_h)
    pesos32 = np.asarray(p["dobrada"]["pesos"], dtype=np.float32)
    if h["n"] != len(pesos32) or not np.array_equal(h["pesos"], pesos32) \
            or h["bias"] != np.float32(p["dobrada"]["bias"]):
        raise SystemExit("Abortado: o lda_modelo.h não é o deste lda_final.json "
                         "(rode o 06 de novo antes).")
    if h["empate_e_falha"] != (p["escore_zero"] == p["classe_positiva"]):
        raise SystemExit("Abortado: a regra de empate do lda_modelo.h difere da do JSON.")
    return h


def _f32(v: float) -> str:
    return H._f32(v)


def gerar_header(p: dict, origem: dict, X: np.ndarray, ref: np.ndarray, falha: np.ndarray,
                 segmento: np.ndarray, gravacao: np.ndarray, tol: np.ndarray,
                 demo_idx: list[int]) -> str:
    n_seg, n_feat = X.shape
    nomes = list(dict.fromkeys(gravacao))
    cod = {g: i for i, g in enumerate(nomes)}
    linhas_x = []
    for i in range(n_seg):
        valores = ", ".join(_f32(v) for v in X[i])
        linhas_x.append(f"    {{ {valores} }},  /* seg {int(segmento[i])} {gravacao[i]} */")

    def lista(fmt, vals, por_linha=8):
        itens = [fmt(v) for v in vals]
        return [("    " + ", ".join(itens[k:k + por_linha]) + ",")
                for k in range(0, len(itens), por_linha)]

    texto = "\n".join([
        "/*",
        " * lda_vetores_teste.h -- vetores de teste da LDA para a placa (marco de 13/10)",
        " *",
        " * GERADO por scripts/pipeline/07_export_lda_test_vectors.py; nao editar a mao.",
        f" * Modelo:  {origem['json']} ({p.get('id', '?')})",
        f" * SHA-256 do JSON: {origem['sha256_json']}",
        f" * SHA-256 do CSV:  {origem['sha256_csv']}",
        f" * Features: {origem['features']} (commit {origem['features_commit']})",
        f" * Commit do gerador: {origem['commit']}",
        " *",
        " * Caracteristicas do Python (o MFCC ainda nao esta portado) de segmentos do",
        " * treino do modelo final: confere o porte do classificador, nao o desempenho.",
        " * vt_tolerancia[i] = limite do arredondamento em float32 (o mesmo do 06)",
        " * mais o arredondamento do escore de referencia para float32.",
        " */",
        "",
        "#ifndef LDA_VETORES_TESTE_H",
        "#define LDA_VETORES_TESTE_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define VT_N_SEGMENTOS {n_seg}",
        f"#define VT_N_FEATURES {n_feat}",
        f"#define VT_N_GRAVACOES {len(nomes)}",
        f"#define VT_N_DEMO {len(demo_idx)}",
        f"#define VT_MODELO_ID \"{p.get('id', '?')}\"",
        f"#define VT_JSON_SHA256_12 \"{origem['sha256_json'][:12]}\"",
        "/* bias do modelo: o firmware compara com o lda_bias do lda_modelo.h */",
        f"#define VT_LDA_BIAS {_f32(p['dobrada']['bias'])}",
        "",
        "static const char *const vt_nomes_gravacao[VT_N_GRAVACOES] = {",
        *[f"    \"{g}\"," for g in nomes],
        "};",
        "",
        "static const float vt_x[VT_N_SEGMENTOS][VT_N_FEATURES] = {",
        *linhas_x,
        "};",
        "",
        "/* escore do escores_referencia.csv (float64 no Python, aqui em float32) */",
        "static const float vt_escore_ref[VT_N_SEGMENTOS] = {",
        *lista(_f32, ref, 6),
        "};",
        "",
        "static const float vt_tolerancia[VT_N_SEGMENTOS] = {",
        *lista(_f32, tol, 6),
        "};",
        "",
        "/* decisao do CSV: 1 = falha, 0 = normal */",
        "static const uint8_t vt_falha_ref[VT_N_SEGMENTOS] = {",
        *lista(lambda v: str(int(v)), falha, 20),
        "};",
        "",
        "/* indice do segmento no escores_referencia.csv */",
        "static const uint16_t vt_segmento[VT_N_SEGMENTOS] = {",
        *lista(lambda v: str(int(v)), segmento, 16),
        "};",
        "",
        "/* indice em vt_nomes_gravacao */",
        "static const uint8_t vt_gravacao[VT_N_SEGMENTOS] = {",
        *lista(lambda g: str(cod[g]), gravacao, 20),
        "};",
        "",
        "/* demonstracao com os LEDs: por gravacao, o segmento de menor |escore| */",
        "static const uint16_t vt_demo[VT_N_DEMO] = {",
        *lista(str, demo_idx, 16),
        "};",
        "",
        "#endif /* LDA_VETORES_TESTE_H */",
        "",
    ])
    if not texto.isascii():
        raise ValueError("o .h tem caracteres fora do ASCII (nome de gravação no CSV?)")
    return texto


def ler_vetores(texto: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lê de volta X, o escore de referência e a tolerância, em float32."""
    import re

    def bloco(nome):
        corpo = re.search(rf"{nome}\[[^\]]*\](?:\[[^\]]*\])? = \{{(.*?)\n\}};", texto, re.S).group(1)
        corpo = re.sub(r"/\*.*?\*/", "", corpo, flags=re.S).replace("{", "").replace("}", "")
        return np.array([np.float32(v.strip().rstrip("f")) for v in corpo.split(",") if v.strip()],
                        dtype=np.float32)

    n = int(re.search(r"#define VT_N_SEGMENTOS (\d+)", texto).group(1))
    m = int(re.search(r"#define VT_N_FEATURES (\d+)", texto).group(1))
    return bloco("vt_x").reshape(n, m), bloco("vt_escore_ref"), bloco("vt_tolerancia")


def conferir(texto_vt: str, texto_lda: str, h: dict, falha: np.ndarray) -> dict:
    """Escore em float32 (Python e, se houver, C do host) a partir do que foi gravado no .h."""
    X32, ref32, tol32 = ler_vetores(texto_vt)
    s_py = H.escore_float32(h, X32)
    dif = np.abs(s_py.astype(float) - ref32.astype(float))
    dec_py = s_py >= 0 if h["empate_e_falha"] else s_py > 0
    out = {
        "n_segmentos": len(ref32),
        "max_dif_escore_f32": float(dif.max()),
        "max_dif_sobre_tol_f32": float((dif / tol32).max()),
        "decisoes_iguais_f32": bool(np.array_equal(dec_py.astype(int), falha.astype(int))),
    }
    cc = H.compilador_c()
    if cc:
        s_c, d_c = H.escore_em_c(texto_lda, X32, cc)
        dif_c = np.abs(s_c - ref32.astype(float))
        out["max_dif_escore_c"] = float(dif_c.max())
        out["max_dif_sobre_tol_c"] = float((dif_c / tol32).max())
        out["decisoes_iguais_c"] = bool(np.array_equal(d_c, falha.astype(int)))
    out["compilador_c"] = H.versao_compilador(cc)
    return out


def aprovada(c: dict) -> bool:
    ok = c["decisoes_iguais_f32"] and c["max_dif_sobre_tol_f32"] <= 1.0
    if "max_dif_sobre_tol_c" in c:
        ok = ok and c["decisoes_iguais_c"] and c["max_dif_sobre_tol_c"] <= 1.0
    return ok


def checar_commit(out: Path, sujos: list[str]) -> tuple[bool, str]:
    """
    (abortar, mensagem). Gravando no `.h` versionado, mudanças não commitadas no
    gerador ou nas entradas abortam; em outro caminho, viram só um aviso.
    """
    if not sujos:
        return False, ""
    lista = "\n  " + "\n  ".join(sujos)
    if Path(out).resolve() == OUT_VERSIONADO.resolve():
        return True, "Abortado: commit antes de gerar; há mudanças não commitadas em:" + lista
    return False, ("aviso: mudanças não commitadas (o commit no .h fica marcado com "
                   "+mudancas):" + lista)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", type=Path, default=Path("reports/modelo_final/lda_final.json"))
    ap.add_argument("--lda-h", type=Path, default=Path("reports/modelo_final/lda_modelo.h"))
    ap.add_argument("--escores", type=Path, default=Path("reports/modelo_final/escores_referencia.csv"))
    ap.add_argument("--features", type=Path,
                    default=Path(config.dir_features(config.FS_TRABALHO)) / "mfcc_features.npz")
    ap.add_argument("--out", type=Path, default=OUT_VERSIONADO)
    ap.add_argument("--por-gravacao", type=int, default=0,
                    help="segmentos por gravação, igualmente espaçados (0 = todos)")
    args = ap.parse_args()

    sujos = H.nao_commitados([Path(__file__), args.modelo, args.escores, args.lda_h])
    abortar, msg = checar_commit(args.out, sujos)
    if msg:
        print(msg)
    if abortar:
        return 1

    p = json.loads(args.modelo.read_text(encoding="utf-8"))
    H.validar_modelo(p)
    texto_lda = args.lda_h.read_text(encoding="utf-8")
    h = conferir_lda_h(texto_lda, p)

    X, ref, prev = H.carregar_referencia(args.features, args.escores, p)
    linhas = list(csv.DictReader(args.escores.open(encoding="utf-8")))
    segmento = np.array([int(l["segmento"]) for l in linhas])
    gravacao = np.array([l["gravacao"] for l in linhas])
    falha = prev == "falha"

    sel = selecionar(gravacao, args.por_gravacao)
    X, ref, falha, segmento, gravacao = X[sel], ref[sel], falha[sel], segmento[sel], gravacao[sel]
    tol = tolerancia(p, X, ref)
    demo_idx = demo(gravacao, ref)

    manifesto = json.loads(args.features.with_name("manifest_features.json").read_text(encoding="utf-8"))
    origem = {
        "json": args.modelo.as_posix(),
        "sha256_json": hashlib.sha256(args.modelo.read_bytes()).hexdigest(),
        "sha256_csv": hashlib.sha256(args.escores.read_bytes()).hexdigest(),
        "features": args.features.as_posix(),
        "features_commit": manifesto.get("git_commit", "?"),
        "commit": experimentos.git_short_hash() + ("+mudancas" if sujos else ""),
    }
    texto = gerar_header(p, origem, X, ref, falha, segmento, gravacao, tol, demo_idx)

    c = conferir(texto, texto_lda, h, falha)
    print(f"modelo {p.get('id')}: {len(sel)} segmentos exportados de {len(linhas)}, "
          f"{X.shape[1]} características, demonstração com {len(demo_idx)}")
    print("conferência: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    if not aprovada(c):
        print("Abortado: o escore em float32 não reproduz a referência dentro da tolerância.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(texto, encoding="utf-8", newline="\n")
    print(f"cabeçalho: {args.out} ({args.out.stat().st_size / 1024:.1f} KB de texto; "
          f"~{(X.size + 2 * len(ref)) * 4 / 1024:.1f} KB de flash)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
