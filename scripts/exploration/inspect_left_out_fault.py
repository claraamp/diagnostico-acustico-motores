#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_left_out_fault.py — onde cai, no espaço de features, a falha deixada de fora
=====================================================================================

Pergunta que este estudo responde: por que a `bpfo_0.3mm` não é detectada no
Protocolo B, com ou sem aumento de dados? Mede a posição da falha deixada de
fora em relação ao normal e às falhas de treino, por duas métricas diferentes,
que respondem a perguntas diferentes e por isso dão números diferentes.

Métricas (todas na mesma convenção: 0 = falhas de treino, 1 = normal)
---------------------------------------------------------------------
1. Distâncias entre centróides, no espaço das 26 features padronizadas
   (z-score com média e desvio de todos os segmentos). Descritiva: não usa
   nenhum classificador. Inclui o raio médio de uma classe, para escala.

2. Posição na reta dos centróides. Projeção do centróide da falha deixada de
   fora na reta que vai do centróide das falhas de treino (0) ao centróide do
   normal (1), no mesmo espaço padronizado. Diz onde a falha fica AO LONGO da
   direção média falha→normal, ignorando o quanto ela se afasta dessa reta.

3. Posição no eixo da LDA. Em cada fold do Protocolo B com essa falha no
   teste, a LDA do `run_protocol` (StandardScaler + LDA, priors iguais) é
   treinada no treino do fold, e o escore dela (`decision_function`) é
   reescalado: 0 = média dos escores das falhas de treino, 1 = média do normal
   de treino. É a direção que o classificador de fato usa para decidir. Com
   `--aumento`, a LDA é treinada com as variantes aceitas no fold, como no
   `run_protocol --aumento`.

A métrica 2 é geométrica e independe do treino; a 3 depende do que a LDA
aprendeu e pondera cada feature pela covariância. Um valor perto de 0,5 na
métrica 2 e perto de 0,8 na 3 não se contradizem: a falha está a meio caminho
na direção média, mas do lado do normal na direção que a LDA escolheu.
(Quem mede a partir do normal, com 0 = normal, obtém 1 − esses valores.)

Uso
---
    python scripts/exploration/inspect_left_out_fault.py
    python scripts/exploration/inspect_left_out_fault.py --falha bpfi_0.3mm
    python scripts/exploration/inspect_left_out_fault.py --aumento   # requer 04 --aumento N

Lê as features do `04_extract_features.py` (e o aumento, se pedido). Não
escreve em data/; grava o resultado em reports/exploration/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np

import config
from augmentation import variantes
from validation import particao
from validation.run_protocol import montar_treino, novo_modelo, rotulos


def posicao(v, zero, um) -> float:
    """Projeção de `v` na reta zero→um, com 0 em `zero` e 1 em `um`."""
    d = um - zero
    return float((v - zero) @ d / (d @ d))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--falha", default="bpfo_0.3mm", choices=config.CLASSES[1:])
    ap.add_argument("--aumento", action="store_true",
                    help="treina a LDA com as variantes aceitas do mfcc_aumento.npz")
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--features", type=Path,
                    default=Path("data/processed/features/mfcc_features.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/exploration"))
    args = ap.parse_args()

    particoes = particao.carregar(args.splits)
    segmentos = particao.segmentos_de(particoes)
    manifesto = json.loads(args.features.with_name("manifest_features.json")
                           .read_text(encoding="utf-8"))
    if manifesto.get("splits_hash") != particao.hash_arquivo(args.splits):
        print("Abortado: features de outra partição; rode o 04 de novo.")
        return 1
    X = np.load(args.features)["X"]
    rot = np.array([s.rotulo for s in segmentos])
    y = rotulos(segmentos, "binario")
    falhas_treino = [c for c in config.CLASSES[1:] if c != args.falha]

    # ------------------------------------------------ 1 e 2: geometria
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    centro = {c: Z[rot == c].mean(axis=0) for c in config.CLASSES}
    raio = float(np.mean([np.linalg.norm(Z[rot == c] - centro[c], axis=1).mean()
                          for c in config.CLASSES]))
    distancias = {c: float(np.linalg.norm(centro[args.falha] - centro[c]))
                  for c in config.CLASSES if c != args.falha}
    centro_falhas = np.mean([centro[c] for c in falhas_treino], axis=0)
    pos_reta = posicao(centro[args.falha], centro_falhas, centro["normal"])

    # ------------------------------------------------ 3: eixo da LDA, por fold
    aumento = None
    if args.aumento:
        info = manifesto.get("aumento")
        caminho = args.features.with_name("mfcc_aumento.npz")
        if not info or not caminho.exists():
            print("Abortado: --aumento pedido, mas não há mfcc_aumento.npz; rode o 04 com --aumento N.")
            return 1
        d = np.load(caminho)
        aumento = {"X": d["X"], "origem": d["origem"],
                   "lidos": variantes.ids_lidos(segmentos, d["origem"],
                                                d["inicio_lido"], d["fim_lido"])}

    por_fold = []
    for f in particao.folds_de(particoes, "B"):
        if f.info.get("falha_de_fora") != args.falha:
            continue
        X_fit, y_fit, _ = montar_treino(f, X, y[f.treino], aumento)
        mdl = novo_modelo("lda", 2).fit(X_fit, y_fit)
        # decision_function > 0 favorece classes_[1]; orienta para "normal" > 0
        s = mdl.decision_function(X)
        if list(mdl.classes_).index("normal") == 0:
            s = -s
        treino = np.zeros(len(segmentos), dtype=bool)
        treino[f.treino] = True
        s_normal = s[treino & (rot == "normal")].mean()
        s_falhas = s[treino & np.isin(rot, falhas_treino)].mean()
        s_fora = s[rot == args.falha]
        por_fold.append({
            "fold": f.nome,
            "posicao_eixo_lda": float((s_fora.mean() - s_falhas) / (s_normal - s_falhas)),
            "fracao_no_lado_normal": float((s_fora > 0).mean()),
        })

    resultado = {
        "falha_deixada_de_fora": args.falha,
        "features": args.features.name,
        "features_commit": manifesto.get("git_commit"),
        "splits": manifesto.get("splits_hash"),
        "aumento": manifesto.get("aumento") if args.aumento else None,
        "convencao": "0 = falhas de treino, 1 = normal",
        "raio_medio_classe_dp": raio,
        "distancia_centroides_dp": distancias,
        "posicao_reta_centroides": pos_reta,
        "posicao_eixo_lda_media": float(np.mean([p["posicao_eixo_lda"] for p in por_fold])),
        "posicao_eixo_lda_min_max": [min(p["posicao_eixo_lda"] for p in por_fold),
                                     max(p["posicao_eixo_lda"] for p in por_fold)],
        "fracao_no_lado_normal_media": float(np.mean([p["fracao_no_lado_normal"]
                                                      for p in por_fold])),
        "por_fold": por_fold,
    }

    print(f"\nFalha deixada de fora: {args.falha}  (convenção: 0 = falhas de treino, 1 = normal)")
    print(f"  raio médio de uma classe: {raio:.2f} dp")
    for c, dist in distancias.items():
        print(f"  distância ao centróide de {c:<11}: {dist:.2f} dp")
    print(f"  [2] posição na reta dos centróides: {pos_reta:.2f}")
    print(f"  [3] posição no eixo da LDA{' (com aumento)' if args.aumento else ''}: "
          f"{resultado['posicao_eixo_lda_media']:.2f} "
          f"(folds: {resultado['posicao_eixo_lda_min_max'][0]:.2f}–"
          f"{resultado['posicao_eixo_lda_min_max'][1]:.2f}); "
          f"no lado 'normal': {resultado['fracao_no_lado_normal_media']:.0%}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    destino = args.out_dir / (f"left_out_{args.falha}" + ("_aumento" if args.aumento else "") + ".json")
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nresultado: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
