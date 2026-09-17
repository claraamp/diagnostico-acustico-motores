import scipy.io as sio
import numpy as np

def to_str(arr):
    arr = np.atleast_1d(arr).ravel()
    return ''.join(str(x) for x in arr)

path = "data/raw/0Nm_Normal.mat"
data = sio.loadmat(path, struct_as_record=True, squeeze_me=False)
sig = data["Signal"][0, 0]

x = sig["x_values"][0, 0]
y = sig["y_values"][0, 0]
fr = sig["function_record"][0, 0]

fs_increment = float(x["increment"][0, 0])
n_values = int(x["number_of_values"][0, 0])
start = float(x["start_value"][0, 0])

print(f"Nome do canal: {to_str(fr['name'])!r}")
print(f"Unidade do eixo Y: {to_str(y['quantity'][0, 0]['label'])!r}")
print(f"Unidade do eixo X: {to_str(x['quantity'][0, 0]['label'])!r}")
print(f"Grupo do canal: {to_str(fr['TL_export_properties_annotation'][0, 0]['channel_group'])!r}")
print(f"Nome do projeto: {to_str(fr['TL_export_properties_annotation'][0, 0]['project_name'])!r}")
print(f"Nome da seção: {to_str(fr['TL_export_properties_annotation'][0, 0]['section_name'])!r}")
print(f"Nome do ensaio: {to_str(fr['TL_export_properties_annotation'][0, 0]['run_name'])!r}")
print()
print(f"start_value: {start}")
print(f"increment (dt): {fs_increment}  ->  fs = {1 / fs_increment:.2f} Hz")
print(f"number_of_values: {n_values}")
print(f"duração: {n_values * fs_increment:.4f} s")
print()

values = y["values"]
print(f"y_values.values shape: {values.shape}, dtype: {values.dtype}")
print(f"min: {values.min():.6f}, max: {values.max():.6f}, mean: {values.mean():.6f}")