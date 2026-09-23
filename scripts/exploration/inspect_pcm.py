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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

import pcm_io

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXCERPT_MS = 120.0   # trecho mostrado na figura

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

    clipes = pcm_io.carregar_clipes(args.pcm_dir, args.manifest)
    pcm_io.relatar_integridade(pcm_io.verificar_integridade(clipes))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1 e 2 — tabela
    # ---------------------------------------------------------------- #
    print(f"\n{len(clipes)} clipes em {args.pcm_dir}\n")

    med = {}
    for c in clipes:
        x = c.x
        med[c.rotulo] = {
            "tipo": c.binario,
            "n": c.n,
            "dur": c.duracao_s,
            "pico_pcm": int(np.max(np.abs(c.pcm))),
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
    fs = clipes[0].fs
    n_exc = int(EXCERPT_MS / 1000 * fs)
    ini = int(5.0 * fs)          # pula os 5 s iniciais (transiente de partida)

    # com clipes curtos (dados truncados ou de teste) o trecho pediria amostras
    # que não existem, e o max() abaixo quebraria num array vazio
    n_min = min(c.n for c in clipes)
    if ini + n_exc > n_min:
        ini = max(0, n_min - n_exc)
        n_exc = min(n_exc, n_min)
        print(f"\n  (clipes curtos: trecho da figura deslocado para t = {ini/fs:.2f} s)")

    ymax = max(float(np.max(np.abs(c.x[ini:ini + n_exc]))) for c in clipes) * 1.1

    fig, axes = plt.subplots(len(clipes), 1, figsize=(10, 1.6 * len(clipes)), sharex=True)
    t = np.arange(n_exc) / fs * 1000
    for ax, c in zip(np.atleast_1d(axes), clipes):
        ax.plot(t, c.x[ini:ini + n_exc], lw=0.6, color="#1f4e79")
        ax.set_ylim(-ymax, ymax)             # MESMA escala em todos: comparação honesta
        ax.set_ylabel(c.rotulo, fontsize=7)
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[-1].set_xlabel("tempo (ms)")
    np.atleast_1d(axes)[0].set_title(
        f"Formas de onda — {n_exc/fs*1000:.0f} ms a partir de t = {ini/fs:.1f} s "
        "(mesma escala vertical)")
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