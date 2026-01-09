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

from datapipes.plotting import plots, map01

def torch_float_1hw_to_np_uint8_hw3(frame: torch.Tensor) -> torch.Tensor:
    frame = einops.repeat(frame, "c h w -> (c 3) h w")
    frame = einops.rearrange(frame, "c h w -> h w c")
    frame = Ops.float01_to_bytes_cpu(map01(frame))
    frame = np.ascontiguousarray(frame.cpu().numpy(), dtype=np.uint8)
    return frame

def create_detector(num_hands: int=2):
    with import_resource.as_path("hand_landmarker.task") as model_path:
        base_options = python.BaseOptions(model_asset_path=str(model_path.as_posix()))
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=num_hands)
        detector = vision.HandLandmarker.create_from_options(options)
        return detector

detector = create_detector(num_hands=2)



def detect_landmarks(img_data: torch.Tensor) -> torch.Tensor:

    # TODO: Remove
    # img_data = img_data.flip(dims=[2])
    # img_data = einops.rearrange(img_data, "c h w -> c w h")

    # Convert to numpy grayscale RGB (h w c) uint8
    img_data = torch_float_1hw_to_np_uint8_hw3(img_data)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_data)

    detection_result = detector.detect(image)

    # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # plots.plot_raw(einops.rearrange(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR), "h w c -> c h w"), cmap="gray")
    return detection_result


def landmarks_to_tensor(landmarks, hand_idx: int=0) -> torch.Tensor:
    return torch.Tensor([(mark.x, mark.y) for mark in landmarks.hand_landmarks[hand_idx]])


