#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_export_stage_references.py — referências por estágio para o harness em C (B2)
================================================================================

Exporta, para alguns segmentos, a saída em float64 de cada estágio da cadeia
(`dsp.estagios_mfcc`) e o escore da LDA final, mais as tolerâncias do A4
(`dsp.ESTAGIOS_PORTE`). O harness em C do repositório do firmware
(`lab3-firmware-stm32`, `tests/host/harness`) lê esses arquivos, roda os
estágios portados com a CMSIS-DSP no PC e compara estágio a estágio
(docs/contrato_numerico.md, seção "Tolerâncias por estágio (A4)").

**Formato** (binário little-endian, sem cabeçalho, para o C ler com `fread`):

    reports/c_reference/estagios/
        tolerancias.txt          estagio metrica tol_isolado tol_encadeado
        modelo.txt               id, SHA-256 do JSON (12) e bias da LDA final
        segmentos.txt            uma pasta de segmento por linha
        manifest.json            commit, hashes das entradas e de cada arquivo
        <rotulo>_seg<id>/
            indice.txt           estagio tipo linhas colunas arquivo
            entrada.i16          o segmento int16 (12.800 amostras)
            escala.f64 … escore.f64

**Segmentos.** Por padrão, o do meio de cada gravação (um por classe), escolhido
pela posição, sem olhar o sinal nem o escore; `--segmentos` escolhe outros ids.

**Conferências antes de gravar** (aborta se alguma falhar): PCM e partição
íntegros; o MFCC dos estágios é o do `dsp.mfcc` usado pelo `04`; o escore de cada
segmento é o do `escores_referencia.csv` (até os 9 dígitos do CSV).

**Commit antes de gerar.** O `manifest.json` grava o commit; gravando no caminho
versionado, o script aborta se ele mesmo, o `dsp.py`, o `lda_final.json` ou o
`escores_referencia.csv` tiverem mudanças não commitadas. Com `--out` em outro
lugar (teste), só avisa. Não há linha no registry: são fixtures de teste, sem
métrica.

Uso
---
    python scripts/pipeline/08_export_stage_references.py
    python scripts/pipeline/08_export_stage_references.py --segmentos 0 147
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
import dsp
import experimentos
import pcm_io
from validation import particao

_spec = importlib.util.spec_from_file_location(
    "export_lda_header", Path(__file__).resolve().with_name("06_export_lda_header.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

OUT_VERSIONADO = Path("reports/c_reference/estagios")
# estágios gravados, na ordem da cadeia; o tipo de cada arquivo
ARQUIVOS = (("entrada", "i16"), ("escala", "f64"), ("janela", "f64"), ("espectro", "f64"),
            ("mel", "f64"), ("logmel", "f64"), ("mfcc", "f64"), ("media", "f64"),
            ("desvio", "f64"), ("escore", "f64"))
DTYPE = {"i16": "<i2", "f64": "<f8"}


def selecionar(segmentos, ids: list[int] | None) -> list:
    """Os segmentos pedidos, ou o do meio de cada gravação (pela posição, sem olhar o sinal)."""
    if ids:
        por_id = {s.id: s for s in segmentos}
        faltando = [i for i in ids if i not in por_id]
        if faltando:
            raise SystemExit(f"Abortado: segmentos inexistentes na partição: {faltando}")
        return [por_id[i] for i in ids]
    escolhidos = []
    for rotulo in dict.fromkeys(s.rotulo for s in segmentos):
        da_gravacao = [s for s in segmentos if s.rotulo == rotulo]
        escolhidos.append(da_gravacao[len(da_gravacao) // 2])
    return escolhidos


def referencias(seg, pcm: np.ndarray, pesos: np.ndarray, bias: float) -> dict[str, np.ndarray]:
    """Estágios do segmento em float64, mais o escore da LDA (forma dobrada, em float64)."""
    e = dsp.estagios_mfcc(pcm, config.FS_TRABALHO)
    if not np.array_equal(e["mfcc"], dsp.mfcc(pcm.astype(np.float64) / config.INT16_FULL,
                                             config.FS_TRABALHO)):
        raise SystemExit(f"Abortado: o MFCC por estágio do segmento {seg.id} não é o do dsp.mfcc")
    e["escore"] = np.array([np.concatenate([e["media"], e["desvio"]]) @ pesos + bias])
    return e


def gravar_segmento(destino: Path, e: dict[str, np.ndarray], cabecalho: list[str]) -> dict[str, str]:
    """Grava os arquivos e o indice.txt de um segmento; devolve o SHA-256 de cada arquivo."""
    destino.mkdir(parents=True, exist_ok=True)
    linhas = [f"# {c}" for c in cabecalho] + ["# estagio tipo linhas colunas arquivo"]
    shas = {}
    for nome, tipo in ARQUIVOS:
        a = np.atleast_2d(np.asarray(e[nome]))
        arquivo = f"{nome}.{tipo}"
        dados = np.ascontiguousarray(a, dtype=DTYPE[tipo]).tobytes()
        (destino / arquivo).write_bytes(dados)
        shas[arquivo] = hashlib.sha256(dados).hexdigest()
        linhas.append(f"{nome} {tipo} {a.shape[0]} {a.shape[1]} {arquivo}")
    (destino / "indice.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8", newline="\n")
    return shas


def ler_segmento(pasta: Path) -> dict[str, np.ndarray]:
    """Lê de volta uma pasta de segmento (o que o harness em C faz), para conferir."""
    out = {}
    for linha in (pasta / "indice.txt").read_text(encoding="utf-8").splitlines():
        if not linha.strip() or linha.startswith("#"):
            continue
        nome, tipo, n, m, arquivo = linha.split()
        out[nome] = np.fromfile(pasta / arquivo, dtype=DTYPE[tipo]).reshape(int(n), int(m))
    return out


def texto_tolerancias() -> str:
    linhas = ["# tolerancias do porte por estagio (A4, docs/contrato_numerico.md);",
              "# metricas: rel_quadro = por quadro, max|c-r|/max|r|; abs = max|c-r|;",
              "# rel = max|c-r|/|r| elemento a elemento",
              "# estagio metrica tol_isolado tol_encadeado"]
    linhas += [f"{e} {m} {i:.0e} {c:.0e}" for e, m, i, c in dsp.ESTAGIOS_PORTE]
    return "\n".join(linhas) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--modelo", type=Path, default=Path("reports/modelo_final/lda_final.json"))
    ap.add_argument("--escores", type=Path, default=Path("reports/modelo_final/escores_referencia.csv"))
    ap.add_argument("--out", type=Path, default=OUT_VERSIONADO)
    ap.add_argument("--segmentos", type=int, nargs="+", default=None, metavar="ID",
                    help="ids dos segmentos (padrão: o do meio de cada gravação)")
    args = ap.parse_args()

    raiz_scripts = Path(__file__).resolve().parents[1]
    sujos = H.nao_commitados([Path(__file__), raiz_scripts / "dsp.py", args.modelo, args.escores])
    if sujos:
        lista = "\n  " + "\n  ".join(sujos)
        if args.out.resolve() == OUT_VERSIONADO.resolve():
            print("Abortado: commit antes de gerar; há mudanças não commitadas em:" + lista)
            return 1
        print("aviso: mudanças não commitadas (o commit no manifest fica marcado com +mudancas):" + lista)

    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(clipes)
    pcm_io.relatar_integridade(divergencias)
    particoes = particao.carregar(args.splits)
    problemas = (particao.conferir_compatibilidade(particoes, clipes, config.FS_TRABALHO)
                 + particao.verificar(particoes))
    if divergencias or problemas:
        print("Abortado: PCM ou partição inválidos:\n  " + "\n  ".join(problemas))
        return 1
    segmentos = particao.segmentos_de(particoes)
    por_rotulo = {c.rotulo: c for c in clipes}

    p = json.loads(args.modelo.read_text(encoding="utf-8"))
    pesos = np.asarray(p["dobrada"]["pesos"], dtype=float)
    bias = float(p["dobrada"]["bias"])
    csv_escore = {int(l["segmento"]): float(l["escore"])
                  for l in csv.DictReader(args.escores.open(encoding="utf-8"))}

    args.out.mkdir(parents=True, exist_ok=True)
    pastas, arquivos = [], {}
    for seg in selecionar(segmentos, args.segmentos):
        pcm = por_rotulo[seg.rotulo].pcm[seg.inicio:seg.fim]
        e = referencias(seg, pcm, pesos, bias)
        ref_csv = csv_escore[seg.id]
        if abs(e["escore"][0] - ref_csv) > 1e-8 * abs(ref_csv) + 1e-9:
            print(f"Abortado: escore do segmento {seg.id} ({e['escore'][0]:.6f}) difere do CSV "
                  f"({ref_csv}); as features ou o modelo não são os do 05")
            return 1
        pasta = f"{seg.rotulo}_seg{seg.id:04d}"
        cab = [f"segmento {seg.id} gravacao {seg.rotulo} indice {seg.indice} "
               f"amostras {seg.inicio}..{seg.fim - 1} do PCM decimado a {config.FS_TRABALHO} Hz"]
        shas = gravar_segmento(args.out / pasta, e, cab)
        lido = ler_segmento(args.out / pasta)
        for nome, _ in ARQUIVOS:
            if not np.array_equal(lido[nome].ravel(), np.asarray(e[nome]).ravel()):
                print(f"Abortado: {pasta}/{nome} não volta igual do arquivo")
                return 1
        pastas.append(pasta)
        arquivos.update({f"{pasta}/{a}": h for a, h in shas.items()})
        print(f"  {pasta}: escore {e['escore'][0]:.3f}")

    (args.out / "segmentos.txt").write_text("\n".join(pastas) + "\n", encoding="utf-8", newline="\n")
    (args.out / "tolerancias.txt").write_text(texto_tolerancias(), encoding="utf-8", newline="\n")
    sha_json = hashlib.sha256(args.modelo.read_bytes()).hexdigest()
    (args.out / "modelo.txt").write_text(
        f"id {p.get('id', '?')}\njson_sha256_12 {sha_json[:12]}\nbias {H._f32(bias).rstrip('f')}\n",
        encoding="utf-8", newline="\n")
    manifesto = {
        "gerador": "scripts/pipeline/08_export_stage_references.py",
        "commit": experimentos.git_short_hash() + ("+mudancas" if sujos else ""),
        "splits": particao.hash_arquivo(args.splits),
        "modelo": p.get("id"), "modelo_sha256": sha_json,
        "fs_hz": config.FS_TRABALHO, "little_endian": True,
        "segmentos": pastas, "arquivos_sha256": arquivos,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    total = sum((args.out / a).stat().st_size for a in arquivos)
    print(f"referências: {args.out} ({len(pastas)} segmentos, {total / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
