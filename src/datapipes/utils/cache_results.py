#%%
import hickle
from pathlib import Path
from typing import Callable
from datapipes.sic import sic
import hashlib

def sanitize_args(args: str) -> str:
    return args.replace(":", ".").replace("\\", ".").replace("/", ".").replace("..", ".")

def args_to_str(*args, **kwargs) -> str:
    args_str = sanitize_args("_".join(str(arg) for arg in args)) if len(args) > 0 else ""
    kwargs_str = sanitize_args(f"_{'_'.join(f'{k}={v}' for k, v in kwargs.items())}") if len(kwargs) > 0 else ""
    result_str = f"{args_str}{kwargs_str}"
    
    # If the result string is too long, hash it
    if len(result_str) > 0: # 75:
        result_str = hashlib.sha256(result_str.encode()).hexdigest()
    
    return result_str

def cache_result_to(folder_path=R"./cached_results", quiet=False, force_recompute=False) -> Callable:
    if not isinstance(folder_path , Path):
        folder_path = Path(folder_path)
    def cache_result(func):
        """
        Use as a @decorator to cache results of expensive computations or load results if they are already cached
        """
        def wrapper(*args, **kwargs):
            filename = f"{func.__name__}__{args_to_str(*args, **kwargs)}.hdf5"
            cache_path = folder_path / Path(filename)
            if cache_path.exists() and not force_recompute:
                cached = hickle.load(cache_path)
                if not quiet:
                    print(f"Using cached result [{filename}]\nDelete {cache_path.absolute()} to recompute")
                return cached
            else:
                result = func(*args, **kwargs)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                hickle.dump(result, cache_path)
                return result
        return wrapper
    return cache_result



cache_result: Callable = cache_result_to()
cache_result_force_recompute: Callable = cache_result_to(force_recompute=True)

# @cache_result
# def gg(st):
#     print(st)
#     return st

# if __name__ == "__main__":
#     gg("kdjfhkldjhf")
# # %%
