#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_pcm.py — primeiro olhar no áudio convertido, antes de qualquer análise
===============================================================================

Passo 2 do roteiro. Responde, nesta ordem:

  1. A conversão do 01 está sã? (amostras e pico conferem com o manifest versionado)
  2. **Há diferença** mensurável entre cada classe de falha e a condição normal?
  3. **Que natureza** tem essa diferença — impulsiva, tonal, banda larga?
  4. Como isso se parece no tempo?

As perguntas 2 e 3 são independentes, e a versão anterior deste script as
misturava: media impulsividade e concluía sobre existência ("os impactos não
estão audíveis neste registro"). Uma falha pode alterar muito o som sem gerar
impactos — foi o que aconteceu no subconjunto 0 Nm deste dataset, e a conclusão
saiu errada. Agora cada pergunta tem sua resposta, e a implicação de método vem
da combinação das duas.

Por que a impulsividade importa
-------------------------------
Falha de rolamento não gera um tom. Gera uma sequência de *impactos*: a cada
passagem de uma esfera sobre o defeito, a estrutura leva uma martelada e toca
suas ressonâncias, que decaem até a próxima. No sinal no tempo isso aparece
como picos curtos sobre um fundo mais baixo — ou seja, o sinal fica "pontudo".

Duas medidas capturam isso sem precisar de espectro nenhum:

  **Fator de crista** = pico / RMS. Ruído gaussiano fica em torno de 3–4.
  Sinal com impactos sobe bem acima disso.

  **Curtose** (4º momento normalizado). Ruído gaussiano dá ≈ 3, por definição.
  Acima de 3, a distribuição tem caudas pesadas — que é o jeito estatístico de
  dizer "tem picos raros e grandes", isto é, impactos. É a base do kurtogram,
  o método padrão de diagnóstico de rolamento.

Curtose **abaixo** de 3 é informação também, e do tipo oposto: indica sinal
oscilatório contínuo dominante (uma senoide pura vale 1,5). Uma classe de falha
com curtose 2,4 e fator de crista menor que o da classe normal não é um sinal
"sem assinatura" — é um sinal cuja diferença é tonal, e que pede análise
espectral em vez de demodulação de envelope.

O que estas medidas NÃO dizem: elas são estatísticas de tempo, cegas para onde
a energia está em frequência. Servem para escolher o próximo instrumento, nunca
para concluir que há ou não há diferença entre classes.

Uso
---
    python scripts/exploration/inspect_pcm.py
    python scripts/exploration/inspect_pcm.py --pcm-dir data/processed/pcm_raw_2Nm

Mexa à vontade: mude `EXCERPT_MS` para ver janelas mais longas ou mais curtas,
ou comente a normalização da figura para ver as amplitudes absolutas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 51_200.0        # Hz — taxa do dataset Jung et al.
INT16_FULL = 32767.0
EXCERPT_MS = 120.0   # trecho mostrado na figura


def carregar(pcm_dir: Path, manifest_path: Path):
    """
    Lê o manifest.json do 01 e carrega os .bin como float em [-1, 1].

    Formato do manifest (um dicionário plano, classe → metadados):
        {"normal": {"arquivo_origem": "0Nm_Normal.mat",
                    "rotulo_binario": "normal",
                    "arquivo_pcm": "data/processed/pcm_raw/normal.bin",
                    "fs_hz": 51200, "n_amostras": 3072000, "duracao_s": 60.0,
                    "pico_original_pa": ..., "pico_pcm": 8505}, ...}

    O manifest É versionado e os .bin NÃO (CONVENTIONS.md, seção 1). Isso faz
    dele a referência para conferir se uma reconversão reproduziu a anterior —
    daí a checagem de `n_amostras` e `pico_pcm` abaixo.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    clipes, divergencias = [], []
    for rotulo, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        # o caminho no manifest é relativo à raiz do repositório; se o script
        # for chamado de outro lugar, cai para <pcm_dir>/<nome do arquivo>
        caminho = Path(meta.get("arquivo_pcm", ""))
        if not caminho.exists():
            caminho = pcm_dir / caminho.name
        if not caminho.exists():
            raise FileNotFoundError(
                f"não achei {caminho}.\n"
                "Os .bin não são versionados: depois de clonar o repositório é preciso "
                "rodar 01_convert_mat_to_pcm.py de novo.")

        fs = float(meta.get("fs_hz", FS))
        if fs != FS:
            raise ValueError(f"{rotulo}: manifest diz fs = {fs} Hz, esperado {FS:.0f} Hz")

        # int16 little-endian, sem cabeçalho — é assim que o 01 grava
        cru = np.fromfile(caminho, dtype="<i2")

        n_esperado = meta.get("n_amostras")
        if n_esperado is not None and cru.size != n_esperado:
            divergencias.append(
                f"{rotulo}: {cru.size} amostras no arquivo, {n_esperado} no manifest")
        p_esperado = meta.get("pico_pcm")
        p_real = int(np.max(np.abs(cru))) if cru.size else 0
        if p_esperado is not None and p_real != p_esperado:
            divergencias.append(
                f"{rotulo}: pico PCM {p_real}, manifest diz {p_esperado}")

        clipes.append({
            "rotulo": str(rotulo),
            "binario": str(meta.get("rotulo_binario", "falha")),
            "origem": meta.get("arquivo_origem"),
            "pcm": cru,                                  # inteiro, como no arquivo
            "x": cru.astype(np.float64) / INT16_FULL,    # float, para as contas
        })

    if divergencias:
        print("\n  ATENÇÃO — os .bin não batem com o manifest versionado:")
        for d in divergencias:
            print(f"    - {d}")
        print("  A conversão do 01 deveria ser determinística. Entenda a diferença antes")
        print("  de seguir: números gerados a partir daqui não serão comparáveis com os")
        print("  das rodadas anteriores.\n")
    else:
        print("  .bin conferem com o manifest versionado (amostras e pico PCM).")

    return clipes


# Limiares de "indistinguível" na pergunta 1. Deliberadamente frouxos: o papel
# deles é sinalizar um caso para inspeção, não decidir nada.
LIM_RMS_DB = 1.0        # diferença de nível abaixo da qual não se afirma nada
LIM_CURTOSE = 0.5       # idem para a forma da distribuição

# Faixas de caráter, ancoradas em valores teóricos: senoide pura = 1,5;
# ruído gaussiano = 3; impactos periódicos = bem acima de 3.
K_TONAL = 2.8
K_IMPULSIVO = 3.5


def carater(k: float) -> str:
    """Classifica o caráter do sinal pela curtose, sem julgar se há falha."""
    if k < K_TONAL:
        return "tonal"
    if k > K_IMPULSIVO:
        return "impulsivo"
    return "banda larga"


def curtose(x: np.ndarray) -> float:
    """Curtose de Fisher + 3 (convenção em que ruído gaussiano vale 3)."""
    xc = x - x.mean()
    m2 = np.mean(xc**2)
    m4 = np.mean(xc**4)
    return float(m4 / (m2**2 + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcm-dir", type=Path, default=Path("data/processed/pcm_raw"))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/exploration"))
    args = ap.parse_args()

    manifest = args.manifest or (args.pcm_dir / "manifest.json")
    clipes = carregar(args.pcm_dir, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1 e 2 — tabela
    # ---------------------------------------------------------------- #
    print(f"\n{len(clipes)} clipes em {args.pcm_dir}\n")

    med = {}
    for c in clipes:
        x = c["x"]
        med[c["rotulo"]] = {
            "tipo": c["binario"],
            "n": c["pcm"].size,
            "dur": c["pcm"].size / FS,
            "pico_pcm": int(np.max(np.abs(c["pcm"]))),
            "rms": float(np.sqrt(np.mean(x**2))),
            "crista": float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-30)),
            "curtose": curtose(x),
        }

    ref = [r for r, m in med.items() if m["tipo"] == "normal"]
    rms_ref = float(np.mean([med[r]["rms"] for r in ref])) if ref else None
    k_ref = float(np.mean([med[r]["curtose"] for r in ref])) if ref else None

    print(f"{'classe':<12} {'tipo':<7} {'amostras':>10} {'dur (s)':>8} {'pico PCM':>9} "
          f"{'RMS':>8} {'ΔRMS dB':>8} {'crista':>7} {'curtose':>8}  caráter")
    print("-" * 104)
    for r, m in med.items():
        d_db = (20 * np.log10(m["rms"] / rms_ref) if rms_ref else float("nan"))
        print(f"{r:<12} {m['tipo']:<7} {m['n']:>10} {m['dur']:>8.4f} {m['pico_pcm']:>9} "
              f"{m['rms']:>8.5f} {d_db:>+8.1f} {m['crista']:>7.2f} {m['curtose']:>8.2f}"
              f"  {carater(m['curtose'])}")

    # ---------------------------------------------------------------- #
    # Leitura: DUAS perguntas independentes, respondidas separadamente
    #
    # Versão anterior deste script misturava as duas — media o caráter do
    # sinal (impulsivo?) e concluía sobre a existência da diferença
    # ("assinatura não está audível"). Uma falha pode diferir muito da
    # condição normal sem ser impulsiva, que é exatamente o caso do 0 Nm
    # deste dataset, e a conclusão saía errada.
    # ---------------------------------------------------------------- #
    if rms_ref is None:
        print("\n  (sem classe normal no manifest — não dá para comparar)")
        return 0

    print(f"\nReferência: classe normal com RMS {rms_ref:.5f}, curtose {k_ref:.2f}, "
          f"crista {np.mean([med[r]['crista'] for r in ref]):.2f}\n")

    print("1) Há diferença mensurável em relação à condição normal?\n")
    indistintas = []
    for r, m in med.items():
        if m["tipo"] == "normal":
            continue
        d_db = 20 * np.log10(m["rms"] / rms_ref)
        dk = m["curtose"] - k_ref
        if abs(d_db) < LIM_RMS_DB and abs(dk) < LIM_CURTOSE:
            indistintas.append(r)
            print(f"   {r:<12} ΔRMS {d_db:+.1f} dB, Δcurtose {dk:+.2f}  "
                  f"→ INDISTINGUÍVEL da normal nestas medidas")
        else:
            print(f"   {r:<12} ΔRMS {d_db:+.1f} dB, Δcurtose {dk:+.2f}  → difere")

    print("\n2) Que natureza tem essa diferença?\n")
    for r, m in med.items():
        if m["tipo"] == "normal" or r in indistintas:
            continue
        print(f"   {r:<12} {carater(m['curtose'])} (curtose {m['curtose']:.2f}, "
              f"crista {m['crista']:.2f})")

    # ---------------------------------------------------------------- #
    # O que cada combinação implica para a escolha de método
    # ---------------------------------------------------------------- #
    print("\nImplicações:\n")
    tipos = {carater(m["curtose"]) for r, m in med.items()
             if m["tipo"] != "normal" and r not in indistintas}
    if "impulsivo" in tipos:
        print("   - Há classes impulsivas: o espectro de envelope (demodulação de Hilbert")
        print("     numa banda ressonante) é o instrumento adequado para BPFI/BPFO.")
    if tipos & {"tonal", "banda larga"}:
        print("   - Há classes NÃO impulsivas: nelas o envelope mede uma estrutura que o")
        print("     sinal não tem em destaque. A diferença existe e é mensurável, mas está")
        print("     na distribuição de energia em frequência — olhe a PSD por classe antes")
        print("     de escolher o critério de validação.")
    if indistintas:
        print(f"   - {', '.join(indistintas)}: sem diferença detectável nestas estatísticas")
        print("     de tempo. Isso NÃO significa ausência de diferença — significa que ela,")
        print("     se existir, está no espectro e não na amplitude ou na impulsividade.")
        print("     É o caso que decide o valor prático do detector, e merece verificação")
        print("     espectral dedicada.")
    print("\n   Nenhuma destas medidas é conclusiva sozinha: são estatísticas de tempo,")
    print("   cegas para onde a energia está em frequência. Elas dizem por onde começar.")

    # ---------------------------------------------------------------- #
    # 3 — figura: mesmo trecho, mesma escala, uma classe por linha
    # ---------------------------------------------------------------- #
    n_exc = int(EXCERPT_MS / 1000 * FS)
    ini = int(5.0 * FS)                      # pula os 5 s iniciais (transiente de partida)
    ymax = max(float(np.max(np.abs(c["x"][ini:ini + n_exc]))) for c in clipes) * 1.1

    fig, axes = plt.subplots(len(clipes), 1, figsize=(10, 1.6 * len(clipes)), sharex=True)
    t = np.arange(n_exc) / FS * 1000
    for ax, c in zip(np.atleast_1d(axes), clipes):
        ax.plot(t, c["x"][ini:ini + n_exc], lw=0.6, color="#1f4e79")
        ax.set_ylim(-ymax, ymax)             # MESMA escala em todos: comparação honesta
        ax.set_ylabel(c["rotulo"], fontsize=7)
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[-1].set_xlabel("tempo (ms)")
    np.atleast_1d(axes)[0].set_title(
        f"Formas de onda — {EXCERPT_MS:.0f} ms a partir de t = 5 s (mesma escala vertical)")
    plt.tight_layout()
    saida = args.out_dir / "fig_formas_de_onda.png"
    plt.savefig(saida, dpi=140)
    plt.close()
    print(f"\nfigura: {saida}")
    print("Na figura: picos curtos, regulares e espaçados, com decaimento entre eles, são")
    print("impactos — dá para contar o espaçamento e comparar com 1/BPFI (3,7 ms) ou 1/BPFO")
    print("(5,6 ms). Oscilação densa e contínua é conteúdo tonal. Faixa uniforme e fina,")
    print("com a mesma altura da classe normal, é o caso indistinguível.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())