"""
selecao.py — validação interna ao treino do Protocolo B e regra de escolha.

Usado por todo estudo que escolhe algo (modelo, conjunto de características) sem
olhar o teste do B: o `compare_classifiers.py` (escolha do classificador) e o
`inspect_comb_features.py` (pente harmônico). Movido do `compare_classifiers.py`
sem mudança, para não depender do torch que aquele script importa.

- `folds_internos`: o B repetido dentro do treino de um fold externo; cada falha
  de treino × cada bloco normal de treino, com a mesma faixa de descarte.
- `escolher`: o melhor pela acurácia balanceada interna média; dentro da margem
  (ou da dispersão entre sementes, se maior), fica o mais barato; em custo igual,
  o modelo de referência, se houver.

Este módulo não é executável: é importado.
"""

from __future__ import annotations


def folds_internos(segmentos, treino: list[int], descarte: int) -> list[tuple[list[int], list[int], str]]:
    """B dentro do treino externo: cada falha de treino × cada bloco normal de treino."""
    sub = [segmentos[i] for i in treino]
    falhas = sorted({s.rotulo for s in sub if s.binario == "falha"})
    blocos = sorted({s.bloco for s in sub if s.binario == "normal"})
    saida = []
    for f in falhas:
        for b in blocos:
            teste = [s.id for s in sub if s.rotulo == f or (s.binario == "normal" and s.bloco == b)]
            por_grav: dict[str, list[int]] = {}
            for i in teste:
                por_grav.setdefault(segmentos[i].rotulo, []).append(segmentos[i].indice)
            tr = []
            for s in sub:
                if s.id in teste:
                    continue
                idx = por_grav.get(s.rotulo)
                if idx and min(abs(s.indice - k) for k in idx) <= descarte:
                    continue
                tr.append(s.id)
            saida.append((tr, teste, f))
    return saida


def escolher(medias: dict, custos: dict, margem: float,
             desvios: dict | None = None, pior: dict | None = None,
             referencia: str | None = None) -> tuple[str, list[str]]:
    """
    Devolve (modelo escolhido, candidatos dentro da tolerância).

    `medias`      acc. bal. interna média por modelo
    `custos`      {"macs": ..., "parametros": ...} por modelo (ex.: `custo()` do compare_classifiers)
    `margem`      tolerância mínima em relação ao melhor
    `desvios`     dispersão entre sementes por modelo (0 se ausente)
    `pior`        pior falha interna por modelo, para desempate em custo igual
    `referencia`  modelo atual (ex.: "lda"): em custo igual, fica ele, e um rival de
                  mesmo custo só o troca se ganhar por mais que a tolerância
    """
    desvios = desvios or {}
    pior = pior or {}
    melhor = max(medias, key=lambda m: (medias[m], m))       # m só para ser determinístico
    cand = [m for m in medias
            if medias[melhor] - medias[m]
            <= max(margem, desvios.get(melhor, 0.0) + desvios.get(m, 0.0)) + 1e-12]
    cand.sort(key=lambda m: (custos[m]["macs"], custos[m]["parametros"], m != referencia,
                             -pior.get(m, 0.0), m))
    return cand[0], cand
