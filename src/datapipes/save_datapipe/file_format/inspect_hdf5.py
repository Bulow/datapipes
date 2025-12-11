import h5py
from rich import print
import numpy as np
from pathlib import Path

def _is_string_dtype(dt):
    # bytes (S), unicode (U), or object that may contain strings
    return dt.kind in ("S", "U") or dt.kind == "O"

_icons = dict(
    # root = "↪",
    root = "↪",
    group = "📁",
    # file = "📄",
    file = "💾",
    tensor = "🧊",
    boolean = "🔀",
    string = "🔤",
    number = "🔢",
    attribute = "🏷️",
)

def _new_tree():
    tree_lines = []
    def print_tree(name, obj):
        def print(s):
            tree_lines.append(str(s))

        depth = name.count('/') + 1
        indent = '  ' * depth
        label = name.split('/')[-1] or '/'
        label = f"{label}"

        if isinstance(obj, h5py.Group):
            print(f"{indent}{_icons["group"]} {label}")

        elif isinstance(obj, h5py.Dataset):
            # Scalar dataset (0-d) — print the value directly for strings & numbers
            if obj.shape == ():
                val = obj[()]
                # Decode bytes to str when needed
                if isinstance(val, (bytes, np.bytes_)):
                    try:
                        val = val.decode('utf-8')
                    except Exception:
                        val = val.decode(errors='replace')
                # String scalar
                if _is_string_dtype(obj.dtype) or isinstance(val, str):
                    print(f"{indent}{_icons["string"]} {label}:  [italic]\"{val}\"[/italic]")
                # Numeric scalar
                elif np.issubdtype(obj.dtype, np.number):
                    # Ensure Python scalar for clean printing
                    try:
                        val = val.item()
                    except Exception:
                        pass
                    print(f"{indent}{_icons["number"]} {label}: [italic]{val}[/italic]")
                # Boolean scalar
                elif np.issubdtype(obj.dtype, np.bool_):
                    print(f"{indent}{_icons["boolean"]} {label}: [italic]{bool(val)}[/italic]")
                else:
                    # Fallback for other scalar dtypes
                    print(f"{indent}{_icons["file"]} {label}  [italic](value: {val}, dtype={obj.dtype})[/italic]")
            else:
                # Non-scalar (arrays)
                if _is_string_dtype(obj.dtype):
                    data = obj[()]
                    # If it's an array of bytes, present as str for readability
                    if isinstance(data, np.ndarray) and data.dtype.kind == 'S':
                        data = data.astype(str)
                    print(f"{indent}{_icons["string"]} {label}:  [italic](string data: {data})[/italic]")
                else:
                    print(f"{indent}{_icons["tensor"]} {label}  [italic](shape={obj.shape}, dtype={obj.dtype})[/italic]")

        # Print attributes (if any)
        if obj.attrs:
            for key, value in obj.attrs.items():
                # Try to pretty-print bytes attributes as utf-8 strings
                if isinstance(value, (bytes, np.bytes_)):
                    try:
                        value = value.decode('utf-8')
                    except Exception:
                        value = value.decode(errors='replace')
                print(f"{indent}{_icons["attribute"]} {key}: {value}")
    return print_tree, tree_lines

def visualize_structure(f: h5py.File, filename: str="/"):
    print_tree, tree_lines = _new_tree()
    tree_lines.append(f"{_icons["root"]} {filename}")
    f.visititems(print_tree)
    # print("\n".join(tree_lines))
    s = "\n".join(tree_lines)
    print(s)

def visualize_file_structure(path: str|Path):
    if isinstance(path, str):
        path = Path(path)
    with h5py.File(path, "r") as f:
        visualize_structure(f, filename=f"{path.name}")
