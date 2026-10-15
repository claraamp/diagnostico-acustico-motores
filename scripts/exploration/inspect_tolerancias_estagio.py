#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_tolerancias_estagio.py — tolerâncias do porte em C, estágio a estágio (A4)
==================================================================================

Pergunta: quanto o MFCC calculado em float32, como o firmware calcula, se afasta
da referência em float64 do Python em cada estágio, e que tolerância separa esse
erro de arredondamento de um erro de porte?

Emula a cadeia em float32 na ordem do C (somas em sequência, sem BLAS), nos
segmentos da partição, de dois jeitos:

- isolado:   cada estágio recebe a referência do estágio anterior, arredondada
             para float32. Mede o erro do próprio estágio; é o que o harness em C
             confere quando só alguns estágios estão portados.
- encadeado: a cadeia inteira em float32 desde o int16, como na placa. O erro de
             um estágio inclui o dos anteriores.

Cada estágio é medido pela métrica de `dsp.erro_estagio` (ver dsp.ESTAGIOS_PORTE).
A tolerância proposta é 10 × o maior erro medido, arredondada para cima na
sequência 1, 2, 5; a encadeada nunca fica abaixo da isolada. O script confere que
`dsp.ESTAGIOS_PORTE` é essa proposta e avisa se não for.

Emulação: escala por divisão; janela com cos em float32; FFT de precisão simples
do scipy (o algoritmo não é o da CMSIS; o harness em C mede a CMSIS de verdade);
Mel e DCT por soma em sequência; média e desvio por Welford
(`dsp.media_desvio_float32`); escore na ordem do `lda_escore()`. Não emula a
fusão de multiplicação e soma (FMA) do Cortex-M4.

Saída: reports/exploration/tolerancias_estagio.json (com --sem-registro, só imprime)

Uso
---
    python scripts/exploration/inspect_tolerancias_estagio.py --responsavel <nome>
    python scripts/exploration/inspect_tolerancias_estagio.py --sem-registro
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
from scipy.fft import dct, rfft

import config
import dsp
import experimentos
import pcm_io
from validation import particao

F = np.float32
ESTAGIOS = [e[0] for e in dsp.ESTAGIOS_PORTE]
METRICA = {e[0]: e[1] for e in dsp.ESTAGIOS_PORTE}
ANTERIOR = {"escala": "entrada", "janela": "escala", "espectro": "janela", "mel": "espectro",
            "logmel": "mel", "mfcc": "logmel", "media": "mfcc", "desvio": "mfcc",
            "escore": "caracteristicas"}


class CadeiaFloat32:
    """Os estágios da cadeia em float32, na ordem e com as somas que o C faz."""

    def __init__(self, fs: int, pesos: np.ndarray, bias: float):
        n_q = int(round(config.MFCC_WINDOW_MS / 1000 * fs))
        n_p = int(round(config.MFCC_HOP_MS / 1000 * fs))
        self.n_fft = 1 << (n_q - 1).bit_length()
        n_quadros = 1 + (config.amostras_por_segmento(fs) - n_q) // n_p
        self.idx = np.arange(n_q)[None, :] + n_p * np.arange(n_quadros)[:, None]
        self.w = (F(0.5) - F(0.5) * np.cos(F(2 * math.pi / (n_q - 1)) * np.arange(n_q, dtype=F))).astype(F)
        self.fb = dsp.mel_filterbank(fs, self.n_fft, config.MFCC_N_MELS).astype(F)
        self.dct = dct(np.eye(config.MFCC_N_MELS), type=2, axis=0, norm="ortho")[:config.MFCC_N_COEFS].astype(F)
        self.pesos, self.bias = np.asarray(pesos, dtype=F), F(bias)

    @staticmethod
    def _soma_em_sequencia(a: np.ndarray, m: np.ndarray) -> np.ndarray:
        """a (quadros × K) por m (saídas × K): acc += a[:, k]·m[:, k], k = 0…K−1, em float32."""
        acc = np.zeros((a.shape[0], m.shape[0]), dtype=F)
        for k in range(a.shape[1]):
            acc = (acc + a[:, k:k + 1] * m[None, :, k]).astype(F)
        return acc

    def escala(self, pcm):
        return (pcm.astype(F) / F(config.INT16_FULL)).astype(F)

    def janela(self, x):
        return (x.astype(F)[self.idx] * self.w).astype(F)

    def espectro(self, quadros):
        X = rfft(quadros.astype(F), n=self.n_fft, axis=1)
        return (X.real * X.real + X.imag * X.imag).astype(F)

    def mel(self, esp):
        return self._soma_em_sequencia(esp.astype(F), self.fb)

    def logmel(self, mel):
        return np.log(mel.astype(F) + F(1e-10)).astype(F)

    def mfcc(self, lm):
        return self._soma_em_sequencia(lm.astype(F), self.dct)

    def resumo(self, m):
        return dsp.media_desvio_float32(m, "welford")

    def escore(self, x):
        s = self.bias
        for p, v in zip(self.pesos, np.asarray(x, dtype=F)):
            s = F(s + F(p * v))
        return np.array([s])


def medir(segmentos, por_rotulo: dict, cadeia: CadeiaFloat32, pesos, bias) -> dict:
    """Maior erro de cada estágio (métrica do A4), isolado e encadeado, e o segmento onde ocorreu."""
    pior = {modo: {e: (0.0, None) for e in ESTAGIOS} for modo in ("isolado", "encadeado")}

    def anotar(modo, estagio, valor, seg_id):
        if valor > pior[modo][estagio][0]:
            pior[modo][estagio] = (valor, seg_id)

    for s in segmentos:
        r = dsp.estagios_mfcc(por_rotulo[s.rotulo].pcm[s.inicio:s.fim], config.FS_TRABALHO)
        r["caracteristicas"] = np.concatenate([r["media"], r["desvio"]])
        r["escore"] = np.array([r["caracteristicas"] @ pesos + bias])
        # isolado: cada estágio parte da referência do anterior
        iso = {e: getattr(cadeia, e)(r[ANTERIOR[e]]) for e in ("escala", "janela", "espectro",
                                                              "mel", "logmel", "mfcc", "escore")}
        iso["media"], iso["desvio"] = cadeia.resumo(r["mfcc"])
        # encadeado: desde o int16
        enc = {"escala": cadeia.escala(r["entrada"])}
        for e in ("janela", "espectro", "mel", "logmel", "mfcc"):
            enc[e] = getattr(cadeia, e)(enc[ANTERIOR[e]])
        enc["media"], enc["desvio"] = cadeia.resumo(enc["mfcc"])
        enc["escore"] = cadeia.escore(np.concatenate([enc["media"], enc["desvio"]]))
        for modo, calc in (("isolado", iso), ("encadeado", enc)):
            for e in ESTAGIOS:
                anotar(modo, e, dsp.erro_estagio(calc[e], r[e], METRICA[e]), s.id)
    return {modo: {e: {"erro": v, "segmento": sid} for e, (v, sid) in d.items()}
            for modo, d in pior.items()}


def tolerancia_proposta(erro: float) -> float:
    """10 × o erro, arredondado para cima na sequência 1, 2, 5."""
    alvo = 10 * erro
    expo = math.floor(math.log10(alvo))
    for m in (1, 2, 5, 10):
        if m * 10.0 ** expo >= alvo * (1 - 1e-12):
            return float(f"{m * 10.0 ** expo:.0e}")
    raise AssertionError


def propor(medido: dict) -> dict:
    prop = {}
    for e in ESTAGIOS:
        iso = tolerancia_proposta(medido["isolado"][e]["erro"])
        enc = max(iso, tolerancia_proposta(medido["encadeado"][e]["erro"]))
        prop[e] = {"metrica": METRICA[e], "isolado": iso, "encadeado": enc}
    return prop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--modelo", type=Path, default=Path("reports/modelo_final/lda_final.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/exploration/tolerancias_estagio.json"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    ap.add_argument("--sem-registro", action="store_true",
                    help="teste: só imprime; não grava o JSON nem escreve no registry")
    args = ap.parse_args()
    if not args.sem_registro and not args.responsavel:
        ap.error("--responsavel é obrigatório numa rodada registrada")

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
    modelo = json.loads(args.modelo.read_text(encoding="utf-8"))
    pesos = np.asarray(modelo["dobrada"]["pesos"], dtype=float)
    bias = float(modelo["dobrada"]["bias"])

    print(f"{len(segmentos)} segmentos; cadeia em float32, isolada e encadeada")
    medido = medir(segmentos, {c.rotulo: c for c in clipes},
                   CadeiaFloat32(config.FS_TRABALHO, pesos, bias), pesos, bias)
    proposta = propor(medido)
    atual = {e: {"metrica": m, "isolado": i, "encadeado": c} for e, m, i, c in dsp.ESTAGIOS_PORTE}
    print(f"  {'estágio':<9} {'métrica':<11} {'isolado':>10} {'tol':>7} {'encadeado':>10} {'tol':>7}")
    for e in ESTAGIOS:
        print(f"  {e:<9} {METRICA[e]:<11} {medido['isolado'][e]['erro']:>10.2e} "
              f"{proposta[e]['isolado']:>7.0e} {medido['encadeado'][e]['erro']:>10.2e} "
              f"{proposta[e]['encadeado']:>7.0e}")
    confere = proposta == atual
    if not confere:
        print("AVISO: dsp.ESTAGIOS_PORTE difere da proposta acima; atualize-o (e o contrato)")
    if args.sem_registro:
        print("(rodada de teste: nada gravado)")
        return 0

    exp_id = f"exp{experimentos.next_exp_number(args.registry):03d}"
    saida = {"id": exp_id, "splits": particao.hash_arquivo(args.splits), "modelo": modelo.get("id"),
             "n_segmentos": len(segmentos), "regra": "10 × o maior erro, para cima em 1, 2, 5; "
             "encadeada ≥ isolada", "medido": medido, "tolerancias": proposta,
             "estagios_porte_confere": confere}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(saida, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    experimentos.append_registry(args.registry, [{
        "id": exp_id,
        "data": date.today().isoformat(),
        "etapa": "contrato_numerico",
        "script": "exploration/inspect_tolerancias_estagio.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "precisao": "float32", "modos": "isolado+encadeado", "splits": saida["splits"],
            "modelo": saida["modelo"], "fs_hz": config.FS_TRABALHO, "margem": 10,
        }),
        "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
        "metricas": experimentos.kv({
            f"{e}_{m[:3]}": f"{medido[m][e]['erro']:.1e}" for e in ESTAGIOS
            for m in ("isolado", "encadeado")
        }),
        "responsavel": args.responsavel,
        "notas": args.notas or ("erro de cada estágio do MFCC em float32 contra float64, isolado e "
                                "encadeado, e tolerância de cada um (A4 do contrato numérico)"),
    }])
    print(f"resultado: {args.out}\nregistrado: {exp_id} em {args.registry}")
    return 0 if confere else 1


if __name__ == "__main__":
    raise SystemExit(main())
