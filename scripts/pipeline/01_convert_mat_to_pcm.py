import scipy.io as sio
import numpy as np
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/pcm_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# nome do arquivo -> (rótulo original de 5 classes, rótulo binário)
CLASS_FILES = {
    "0Nm_Normal.mat":  ("normal",   "normal"),
    "0Nm_BPFI_03.mat": ("bpfi_0.3mm", "falha"),
    "0Nm_BPFI_10.mat": ("bpfi_1.0mm", "falha"),
    "0Nm_BPFO_03.mat": ("bpfo_0.3mm", "falha"),
    "0Nm_BPFO_10.mat": ("bpfo_1.0mm", "falha"),
}

def load_acoustic(path):
    data = sio.loadmat(path, struct_as_record=True, squeeze_me=False)
    sig = data["Signal"][0, 0]
    x, y, fr = sig["x_values"][0, 0], sig["y_values"][0, 0], sig["function_record"][0, 0]

    grupo = "".join(str(c) for c in np.atleast_1d(fr["TL_export_properties_annotation"][0, 0]["channel_group"]).ravel())
    assert grupo == "Acoustic", f"{path}: esperava canal 'Acoustic', achei {grupo!r}"

    fs = 1.0 / float(x["increment"][0, 0])
    n = int(x["number_of_values"][0, 0])
    values = y["values"].astype(np.float64).ravel()

    assert abs(fs - 51200.0) < 1.0, f"{path}: fs inesperada ({fs} Hz)"
    assert values.shape[0] == n, f"{path}: número de amostras não bate"

    return values, fs

# --- passo 1: carregar os 5 e achar o pico global ---
raw = {}
for fname, (label5, label_bin) in CLASS_FILES.items():
    values, fs = load_acoustic(RAW_DIR / fname)
    raw[fname] = values
    print(f"{fname}: {label5:12s} | dur={len(values)/fs:.4f}s | pico={np.abs(values).max():.6f} Pa")

global_peak = max(np.abs(v).max() for v in raw.values())
scale = 0.98 * 32767 / global_peak  # 2% de margem pra não estourar por arredondamento
print(f"\nPico global entre os 5 arquivos: {global_peak:.6f} Pa -> fator de escala: {scale:.2f}")

# --- passo 2: converter e salvar, com o mesmo fator pra todos ---
manifest = {}
for fname, (label5, label_bin) in CLASS_FILES.items():
    values = raw[fname]
    pcm = np.round(values * scale).astype(np.int16)

    out_path = OUT_DIR / f"{label5}.bin"
    pcm.tofile(out_path)

    manifest[label5] = {
        "arquivo_origem": fname,
        "rotulo_binario": label_bin,
        "arquivo_pcm": str(out_path),
        "fs_hz": 51200,
        "n_amostras": len(pcm),
        "duracao_s": len(pcm) / 51200,
        "pico_original_pa": float(np.abs(values).max()),
        "pico_pcm": int(np.abs(pcm).max()),
    }
    print(f"{fname} -> {out_path.name}  (pico PCM: {np.abs(pcm).max()} / 32767)")

with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nManifesto salvo em {OUT_DIR / 'manifest.json'}")