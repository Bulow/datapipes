from __future__ import annotations
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from datapipes.plotting.torch_colormap import TorchColormap
from datapipes.utils import import_resource
import torch
from datapipes.ops import Ops
import einops
import numpy as np
from typing import Literal, Optional, Dict
from datapipes.plotting import plots, map01
from dataclasses import dataclass

VisionRunningMode = mp.tasks.vision.RunningMode

DetectorType = vision.HandLandmarker

def torch_float_1hw_to_np_uint8_hw3(frame: torch.Tensor) -> torch.Tensor:
    frame = einops.repeat(frame, "c h w -> (c 3) h w")
    frame = einops.rearrange(frame, "c h w -> h w c")
    frame = Ops.float01_to_bytes_cpu(map01(frame))
    frame = np.ascontiguousarray(frame.cpu().numpy(), dtype=np.uint8)
    return frame

def create_detector(num_hands: int=2, mode: Literal["image", "video"]="video") -> Detector:
    with import_resource.as_path("hand_landmarker.task") as model_path:
        base_options = python.BaseOptions(model_asset_path=str(model_path.as_posix()))
        options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO,
                                            num_hands=num_hands)
        detector: vision.HandLandmarker = vision.HandLandmarker.create_from_options(options)
        return detector

# default_detector: DetectorType = create_detector(num_hands=2) # TODO: per dataset instance as detector keeps state


class Detector:
    def __init__(self, fps: float=100, num_hands: int=2):
        self.detector: DetectorType = create_detector(num_hands=num_hands)
        self.fps: float = fps
        self._n_frame: int = 1

        self.previous = None #: Dict[str, torch.Tensor] = {}
        self.first_run: bool = True
        self.raw_marks = []
        self.ema_marks = []

    def next_timestamp(self) -> float:
        ts = int(self._n_frame * (1000 / self.fps)) # ms
        self._n_frame += 1
        # print(f"{ts = :_}")
        return ts
    
    def detect(self, img: torch.Tensor, ema_alpha: Optional[float]=None) -> Dict[str, torch.Tensor]:
        raw_landmarks_mediapipe_fmt = detect_landmarks(img_data=img, detector=self)
        hands_landmarks_px = extract_landmarks(raw_landmarks_mediapipe_fmt=raw_landmarks_mediapipe_fmt, img=img)

        if ema_alpha is not None:
            ema_hands_landmarks_px = self.ema(hands_landmarks_px, alpha=ema_alpha)
            return ema_hands_landmarks_px
        else:
            return hands_landmarks_px
    
    def ema(self, markers: Dict[str, torch.Tensor], alpha: float=0.9) -> Dict[str, torch.Tensor]:
        # if len(self.previous) == 0:
        #     print(f"{len(self.previous) = }, {alpha = }")
        #     self.previous = {k:torch.empty_like(v).copy_(v) for k, v in markers.items()}
        #     return markers

        if self.first_run:
            self.first_run = False
            self.previous = {k:torch.zeros_like(v).copy_(v) for k, v in markers.items()}
            # print(f"{self.previous = }")
        
        out = {}
        for hand_name, marks in markers.items():
            out[hand_name] = (
                (self.previous[hand_name] * alpha)
                + ((1.0 - alpha) * marks)
            )
            self.previous[hand_name].copy_(out[hand_name], non_blocking=False)
        # assert not torch.allclose(markers["Left"], out["Left"])
        # print(f"applied ema: {out["Left"][0] - self.previous["Left"][0]}")
        self.ema_marks.append(out)
        self.raw_marks.append(markers)
        return out

def extract_landmarks(raw_landmarks_mediapipe_fmt: Dict, img: torch.Tensor) -> Dict[str, torch.Tensor]:
    hand_indices = {cat[0].category_name:cat[0].index for cat in raw_landmarks_mediapipe_fmt.handedness}
    hands_landmarks_px = {hand_name.lower():landmarks_to_tensor(raw_landmarks_mediapipe_fmt, img_shape=img.shape, hand_idx=idx, coord_type="px") for hand_name, idx in hand_indices.items()}
    return hands_landmarks_px

def detect_landmarks(img_data: torch.Tensor, detector: Optional[Detector]=None) -> Dict:
    # TODO: Remove
    # img_data = img_data.flip(dims=[2])
    # img_data = einops.rearrange(img_data, "c h w -> c w h")

    if detector is None:
        raise RuntimeError(f"{detector = }")
        detector = Detector(num_hands=2)

    # Convert to numpy grayscale RGB (h w c) uint8
    img_data = torch_float_1hw_to_np_uint8_hw3(img_data)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_data)


    # detection_result = detector.detector.detect_for_video(image, detector.next_timestamp())
    detection_result = detect_with_retries(image=image, detector=detector, n_retries=3)

    

    # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # plots.plot_raw(einops.rearrange(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR), "h w c -> c h w"), cmap="gray")
    return detection_result

def detect_with_retries(image: mp.Image, detector: Detector, n_retries: int=3) -> Dict:
    n_try = 0

    while n_try < n_retries:
        try:
            detection_result = detector.detector.detect_for_video(image, detector.next_timestamp())
            return detection_result

        except KeyError as ex:
            n_try += 1
            if n_try > n_retries:
                raise ex



def landmarks_to_tensor(landmarks, img_shape: torch.Size, hand_idx: int=0, coord_type: Literal["px", "normalized"]="px") -> torch.Tensor:
    # print(f"landmarks_to_tensor >")
    # import rich
    # rich.print(landmarks)

    # # hand_world_landmarks
    normalized_landmarks = torch.Tensor([(mark.x, mark.y) for mark in landmarks.hand_landmarks[hand_idx]])
    match coord_type:
        case "px":
            h, w = img_shape[-2:]
            return normalized_landmarks * torch.tensor((w, h), device=normalized_landmarks.device)
        case "normalized":
            return normalized_landmarks
        case _:
            raise ValueError(f"Unrecognized coord_type: {coord_type}")    



