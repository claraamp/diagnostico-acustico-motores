#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_divisao_hustmotor.py — divisão do HUSTmotor por rotação, num arquivo versionado
===================================================================================

Grava em `data/processed/hustmotor/divisao_rotacao.json` (VERSIONADO, como o
`splits.json` do Jung) o papel de cada um dos 40 segmentos do HUSTmotor em cada
leitura do Protocolo C, para que todos os scripts e todas as rodadas usem a mesma
divisão, com o mesmo identificador de segmento:

- `segmentos`: os 40 segmentos, cada um com um id estável (`<arquivo>_a<aquisição>`,
  por exemplo `h_20hz_a3`), a classe, a rotação nominal, a aquisição (1 a 5) e as
  amostras do .bin a 12,8 kHz que ele ocupa (`pcm_io.cortar_segmentos`).
- `c2`: a divisão do C2 (`divisao_c2` do manifest, decidida em 20/10), rotação a
  rotação. Na normal, as aquisições de calibração, de descarte e de medida; na falha,
  as de medida da falha. É a que o `run_protocol_c.py` usa no D2 do C2 e na
  recalibração da LDA.
- `rotacao_deixada_de_fora`: para cada rotação de teste, a calibração com a normal
  das outras três rotações (todas as aquisições: as rotações são arquivos
  diferentes, então não há vazamento no tempo para descartar) e o teste com os
  normais e as falhas da rotação deixada de fora. `dentro_da_faixa_calibrada` diz se
  a rotação de teste fica entre a menor e a maior rotação da calibração (10 e 20 Hz)
  ou fora delas (5 e 30 Hz, extrapolação de rotação). A regra de leitura dessa divisão
  é a da página da adoção do D2; este arquivo só fixa os segmentos.

O arquivo sai só do manifest versionado: não precisa dos .bin nem dos .txt. Como o
`03`, se o arquivo existente for diferente, o script para em vez de sobrescrever (uma
divisão nova invalida a comparação com as rodadas que usaram a anterior); use
`--sobrescrever` só depois de registrar essa decisão. O hash curto do arquivo
(`hash_divisao`) pode ir para o registry, como o do `splits.json`.

Uso
---
    python scripts/pipeline/11_divisao_hustmotor.py
    python scripts/pipeline/11_divisao_hustmotor.py --conferir     # só confere, não grava
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import config

MANIFEST = Path("data/processed/hustmotor/pcm_decimated/12800/manifest.json")
SAIDA = Path("data/processed/hustmotor/divisao_rotacao.json")


def id_segmento(rotulo: str, aquisicao: int) -> str:
    """Id estável de um segmento do HUST: o arquivo e a aquisição (1 a 5)."""
    return f"{rotulo}_a{aquisicao}"


def gerar(manifest: dict, caminho_manifest: str = str(MANIFEST)) -> dict:
    """A divisão por rotação, a partir do manifest do HUST (sem dados)."""
    conjunto = manifest["_conjunto"]
    regra = conjunto["divisao_c2"]
    n = config.amostras_por_segmento(config.FS_TRABALHO)
    rotulos = sorted((k for k in manifest if not k.startswith("_")),
                     key=lambda r: (manifest[r]["classe_hust"] != "H", manifest[r]["velocidade_nominal_hz"]))
    segmentos = []
    for r in rotulos:
        m = manifest[r]
        for a in range(1, int(m["n_segmentos"]) + 1):
            segmentos.append({"id": id_segmento(r, a), "arquivo": r, "classe": m["classe_hust"],
                              "binario": m["rotulo_binario"], "rotacao": int(m["velocidade_nominal_hz"]),
                              "aquisicao": a, "amostra_inicio": (a - 1) * n, "amostra_fim": a * n})
    rotacoes = sorted({s["rotacao"] for s in segmentos})

    def ids(cond):
        return [s["id"] for s in segmentos if cond(s)]

    c2 = {}
    for v in rotacoes:
        normal = lambda s, v=v: s["rotacao"] == v and s["classe"] == "H"
        falha = lambda s, v=v: s["rotacao"] == v and s["classe"] == "BF"
        c2[str(v)] = {
            "calibracao": ids(lambda s: normal(s) and s["aquisicao"] in regra["blocos_calibracao"]),
            "descarte": ids(lambda s: normal(s) and s["aquisicao"] in regra["blocos_descarte"]),
            "medida_normal": ids(lambda s: normal(s) and s["aquisicao"] in regra["blocos_medida"]),
            "medida_falha": ids(lambda s: falha(s) and s["aquisicao"] in regra["blocos_medida_falha"]),
        }

    fora = {}
    for v in rotacoes:
        outras = [w for w in rotacoes if w != v]
        fora[str(v)] = {
            "rotacoes_calibracao": outras,
            "dentro_da_faixa_calibrada": bool(min(outras) < v < max(outras)),
            "calibracao": ids(lambda s: s["rotacao"] != v and s["classe"] == "H"),
            "teste_normal": ids(lambda s: s["rotacao"] == v and s["classe"] == "H"),
            "teste_falha": ids(lambda s: s["rotacao"] == v and s["classe"] == "BF"),
        }

    return {
        "gerador": "scripts/pipeline/11_divisao_hustmotor.py",
        "fonte": caminho_manifest,
        "fonte_sha256_origem": {r: manifest[r]["sha256_origem"] for r in rotulos},
        "fs_hz": config.FS_TRABALHO,
        "amostras_por_segmento": n,
        "rotacoes": rotacoes,
        "segmentos": segmentos,
        "c2": {"regra": regra, "por_rotacao": c2},
        "rotacao_deixada_de_fora": {
            "regra": "calibração: a normal das outras três rotações, todas as aquisições; teste: a normal e "
                     "a falha da rotação deixada de fora. A regra de leitura é a da página da adoção do D2.",
            "por_rotacao": fora},
    }


def texto(divisao: dict) -> str:
    return json.dumps(divisao, ensure_ascii=False, indent=2) + "\n"


def hash_divisao(caminho: Path = SAIDA) -> str:
    """Hash curto do arquivo, como o do splits.json: pode ir para o registry."""
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()[:10]


def carregar(caminho: Path = SAIDA) -> dict:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"não achei {caminho}. Ele é versionado; se não existe, rode "
                                "scripts/pipeline/11_divisao_hustmotor.py.")
    return json.loads(caminho.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--conferir", action="store_true", help="só confere o arquivo existente, não grava")
    ap.add_argument("--sobrescrever", action="store_true")
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    novo = texto(gerar(manifest, args.manifest.as_posix()))
    if args.saida.exists():
        if args.saida.read_text(encoding="utf-8") == novo:
            print(f"{args.saida} confere com o manifest (hash {hash_divisao(args.saida)})")
            return 0
        if args.conferir or not args.sobrescrever:
            print(f"Abortado: {args.saida} é diferente do que o manifest gera. Uma divisão nova invalida a "
                  "comparação com as rodadas anteriores; use --sobrescrever só depois de registrar a decisão.")
            return 1
    elif args.conferir:
        print(f"Abortado: {args.saida} não existe")
        return 1
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(novo, encoding="utf-8", newline="\n")
    print(f"gravado: {args.saida} (hash {hash_divisao(args.saida)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
