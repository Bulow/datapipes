"""
Plot tensors
"""

from datapipes.utils import introspection
from datapipes.utils.slicer import Slicer
from datapipes.utils.forced_tty_mode_tqdm import tty_tqdm
from datapipes.utils import cache_results

def enable_jupyter_autoreload():
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("load_ext", "autoreload")
        ip.run_line_magic("autoreload", "2")

def print_gpu_info():
    import torch
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")

    # If CUDA is available, print additional info
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available - PyTorch will use CPU")

__all__ = ["introspection", "Slicer", "tty_tqdm", "cache_results", "enable_jupyter_autoreload", "print_gpu_info"]