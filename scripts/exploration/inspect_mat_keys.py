import scipy.io as sio

path = "data/raw/0Nm_Normal.mat"

try:
    data = sio.loadmat(path)
    print("Carregado com scipy.io.loadmat (MAT < v7.3)\n")
    for key, val in data.items():
        if key.startswith("__"):
            continue
        print(f"Variável: {key!r}")
        print(f"  tipo: {type(val)}, shape: {getattr(val, 'shape', None)}, dtype: {getattr(val, 'dtype', None)}")
except NotImplementedError:
    import h5py
    print("Arquivo é MAT v7.3 (HDF5) — usando h5py\n")
    with h5py.File(path, "r") as f:
        def explorar(nome, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {nome!r}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                print(f"Grupo: {nome!r}")
        f.visititems(explorar)