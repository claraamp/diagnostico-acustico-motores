import scipy.io as sio
import numpy as np

path = "data/raw/0Nm_Normal.mat"
data = sio.loadmat(path, struct_as_record=True, squeeze_me=False)

def dump(obj, name="root", depth=0, max_depth=6):
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{name}: (profundidade máxima atingida)")
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:  # struct do MATLAB
            print(f"{indent}{name}: struct, shape={obj.shape}, campos={obj.dtype.names}")
            for idx in np.ndindex(obj.shape):
                elem = obj[idx]
                for field in obj.dtype.names:
                    dump(elem[field], name=f"{name}{list(idx)}.{field}", depth=depth + 1, max_depth=max_depth)
        elif obj.dtype == object:
            print(f"{indent}{name}: object array, shape={obj.shape}")
            for idx in np.ndindex(obj.shape):
                dump(obj[idx], name=f"{name}{list(idx)}", depth=depth + 1, max_depth=max_depth)
        else:
            print(f"{indent}{name}: array numérico, shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{indent}{name}: {type(obj)} = {repr(obj)[:80]}")

dump(data["Signal"], name="Signal")