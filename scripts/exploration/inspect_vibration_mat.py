#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_vibration_mat.py — o que tem dentro de um .mat de vibração do Jung
==========================================================================

O `01_convert_mat_to_pcm.py` sabe ler os .mat de ÁUDIO (estrutura `Signal`,
canal "Acoustic", 51,2 kHz, um canal). Os de VIBRAÇÃO têm os mesmos nomes de
arquivo, mas outro conteúdo: segundo o artigo (Jung et al., 2023, seção 2),
cinco colunas — carimbo de tempo, x e y do mancal A, x e y do mancal B — a
25,6 kHz, em g. Antes de analisar, é preciso ver como isso está guardado.

Este script percorre o arquivo inteiro e imprime a árvore: cada campo, seu
tipo, forma e, quando for texto, o conteúdo. Não interpreta nada.

Uso
---
    python scripts/exploration/inspect_vibration_mat.py data/raw/vibracao/0Nm_Normal.mat
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


def texto(a) -> str | None:
    """Conteúdo de um campo de texto do MATLAB, ou None se não for texto."""
    a = np.asarray(a)
    if a.dtype.kind in ("U", "S"):
        return "".join(str(c) for c in a.ravel())
    return None


def resumo_numerico(a: np.ndarray) -> str:
    s = f"shape={a.shape} dtype={a.dtype}"
    if a.size and a.dtype.kind in "iuf":
        flat = a.ravel()
        s += f"  min={flat.min():.6g} max={flat.max():.6g}"
        if a.size > 1:
            s += f"  primeiros={np.array2string(flat[:4], precision=6)}"
    return s


def percorrer(nome: str, v, nivel: int = 0, max_nivel: int = 8) -> None:
    pad = "  " * nivel
    if nivel > max_nivel:
        print(f"{pad}{nome}: (profundo demais, parei)")
        return
    if isinstance(v, np.ndarray) and v.dtype.names:          # struct do MATLAB
        print(f"{pad}{nome}: struct shape={v.shape} campos={list(v.dtype.names)}")
        el = v.flat[0] if v.size else None
        if el is None:
            return
        for campo in v.dtype.names:
            percorrer(campo, el[campo], nivel + 1, max_nivel)
        return
    if isinstance(v, np.ndarray) and v.dtype == object:      # cell / objeto
        print(f"{pad}{nome}: object shape={v.shape}")
        for i, item in enumerate(v.ravel()[:6]):
            percorrer(f"[{i}]", item, nivel + 1, max_nivel)
        if v.size > 6:
            print(f"{pad}  ... (+{v.size - 6} itens)")
        return
    t = texto(v)
    if t is not None:
        print(f"{pad}{nome}: texto = {t!r}")
        return
    if isinstance(v, np.ndarray):
        print(f"{pad}{nome}: {resumo_numerico(v)}")
        return
    print(f"{pad}{nome}: {type(v).__name__} = {v!r}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.split("Uso")[1])
    caminho = Path(sys.argv[1])
    try:
        data = sio.loadmat(caminho, struct_as_record=True, squeeze_me=False)
        print(f"{caminho}: MAT < v7.3 (scipy)\n")
        for k, v in data.items():
            if not k.startswith("__"):
                percorrer(k, v)
    except NotImplementedError:
        import h5py
        print(f"{caminho}: MAT v7.3 / HDF5 (h5py)\n")
        with h5py.File(caminho, "r") as f:
            def ver(nome, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {nome}: shape={obj.shape} dtype={obj.dtype}")
                else:
                    print(f"  {nome}/")
            f.visititems(ver)


if __name__ == "__main__":
    main()