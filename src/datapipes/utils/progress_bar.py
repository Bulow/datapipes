from tqdm import tqdm
import gradio as gr
from typing import Callable, Any, Iterable, Iterator, Optional, Literal

class ProgressBar:
    def __init__(self):
        self._create_progress_bar: Callable[[Iterable, Optional[int], Optional[str]], Iterator] = tqdm

    def set_progress_bars(self, new_progress_bar_decorator: Callable[[Iterable, Optional[int], Optional[str]], Iterator]):
        self._create_progress_bar = new_progress_bar_decorator

    def progress_bar(self, tasks: Iterable, total: Optional[int], desc: Optional[str]) -> Iterator:
        yield from self._create_progress_bar(tasks, desc=desc, total=total)


def progress_bar(iterable: Iterable, total: Optional[int], desc: Optional[str], progress_bar_type: Literal["tqdm", "gradio"]) -> Iterator:
    match progress_bar_type:
        case "tqdm":
            yield from tqdm(iterable=iterable, total=total, desc=desc)
        case "gradio":
            progress = gr.Progress()
            yield from progress.tqdm(iterable=iterable, total=total, desc=desc)
