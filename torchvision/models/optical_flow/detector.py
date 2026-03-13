import os
import gc
from typing import Optional, List, Tuple, Union, Callable
from shutil import copytree, copyfile
from textwrap import dedent

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch import Tensor
from torchvision.utils import flow_to_image
from torchvision.io import write_jpeg, write_png


def _release_memory(with_no_grad: bool=True):
    """Releases memory. 
    
    Steps:
        - runs garbage collection: ``gc.collect()``
        - empties the gpu cache: ``torch.cuda.empty_cache()``
    """
    gc.collect()
    if torch.cuda.is_available():
        if with_no_grad:
            with torch.no_grad():
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()


@torch.no_grad()
def raft_preprocess(batch: Tensor) -> Tensor:
    """Returns RAFT model preprocessing transformation."""
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    return transforms(batch)


@torch.no_grad()
def _prepare_frame(frame: Union[Tensor, List[Tensor]], preprocess: Optional[Callable]=None) -> Tensor:
    """Prepares a frame and returns the preprocessed frame as a Tensor of shape 
    (N, 3, H, W).

    Args:
        frame (Union[Tensor, List[Tensor]]): Input frame(s).
        .. warning::
            Input ``frame`` should be of shape ``(N, 3, H, W)``, ``(N, 1, H, W)``, 
            ``(3, H, W)``, ``(1, H, W)`` or ``(H, W)``.
        preprocess (Callable, optional): Preprocessing function for the model. 
            By default, this uses the preprocessing function for ``RAFT`` model.

    Returns:
        frame (Tensor): Preprocessed frame of shape (N, 3, H, W).
        
    """
    frame_shape = frame.shape
    if isinstance(frame, list):
        device = frame[0].device
        frame = torch.stack(frame).to(device)
    else:
        device = frame.device
    # Ensure the frame has the shape: (N, C, H, W); C = 3
    if frame.dim() == 3:
        if frame.shape[0] == 1:
            frame = torch.tile(frame, (3, 1, 1))
        # Set shape: (1, 3, H, W); N = 1, C = 3
        frame = frame.unsqueeze(0)
    elif frame.dim() == 2:
        # Set shape: (1, 3, H, W); N = 1, C = 3
        frame = torch.tile(frame.unsqueeze(0), (3, 1, 1)).unsqueeze(0)
    # Assuming C == 3
    if frame.shape[1] != 3 or frame.dim() > 4:
        raise ValueError(dedent(
            f"""Input should be of shape (N, 3, H, W), (N, 1, H, W), 
            (3, H, W), (1, H, W) or (H, W); got {frame_shape} 
            and adjusted to {frame.shape}.
            """))
    if preprocess is None:
        preprocess = raft_preprocess
    return preprocess(frame).to(device)


def detect_flow(frameA: Union[Tensor, List[Tensor]], 
                frameB: Union[Tensor, List[Tensor]], 
                model: Optional[nn.Module]=None, 
                device: Optional[str]=None,
                output_device: str="cpu",
                preprocess: Optional[Callable]=None,
                save_flow_image: bool=False,
                save_flow_data: bool=False,
                output_folder: str="outputs",
                flow_image_folder: str="flowviz",
                flow_data_folder: str="flowraw",
                flow_basename: Optional[str]=None,                 
                flow_id: Optional[Union[int, str]]=None, 
                ext: str="png") -> Tuple[Tensor, Tensor]:
    """
    Detects optical flow of a single image-pair, based on a given model, 
    and saves and/or returns a tuple of an RGB image Tensor of shape 
    (3, H, W) and a flow Tensor of shape (2, H, W).

    Args:
        frameA (Union[Tensor, List[Tensor]]): First frame(s).
        frameB (Union[Tensor, List[Tensor]]): Second frame(s).
            .. info::
                ``frameA`` and ``frameB`` should be of shape:
                ``(N, 3, H, W)``, ``(N, 1, H, W)``, ``(3, H, W)``, 
                ``(1, H, W)`` or ``(H, W)``.
        model (nn.Module, optional): PyTorch model for Optical Flow. 
            (default: ``torchvision.models.optical_flow.raft_large`` model)
        device (str, optional): Device to run optical flow prediction on; 
            e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc. By default, if 
            ``cuda`` is not available, this falls back to ``cpu``.
        output_device (str, optional): Device to return the output tensors on; 
            e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc. 
            (default: ``"cpu"``)
        preprocess (Callable, optional): Preprocessing function for the model. 
            By default, this uses the preprocessing function for ``RAFT`` model.
        save_flow_image (bool): Whether to save the image converted from flow.
            (default: ``False``)
        save_flow_data (bool): Whether to save the flow data on disk.
            (default: ``False``)
        output_folder (str): Folder to save the optical flow images to. 
            (default: ``"outputs"``)
        flow_image_folder (str): Folder to save the optical flow images to. 
            (default: ``"flowviz"``)
        flow_data_folder (str): Folder to save the optical flow data to. The 
            flow data is saved as a ``.pt`` file.
            (default: ``"flowraw"``)
        flow_basename (str, optional): The basename of the flow-image file.
            (default: ``"predicted_flow"``)
        flow_id (int, str, optional): The numeric flow-id to add to the basename 
            of the flow-image file. 
            (default: ``None``)
        ext (str): The extension for flow-image file (``png`` or ``jpg``). 
            (default: ``"png"``)

    Returns:
        flow_img (Tensor): Tensor of shape ``(3, H, W)``.
        predicted_flow (Tensor): Tensor of shape ``(2, H, W)``.

    Usage:

        1. Download video data.

        ```python
        import gc
        import tempfile
        from pathlib import Path
        from urllib.request import urlretrieve


        video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
        video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
        _ = urlretrieve(video_url, video_path)
        gc.collect() # release memory
        ```

        2. Extract the video frames.

        ```python    
        from torchvision.io import read_video

        frames, _, _ = read_video(str(video_path))
        frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        gc.collect() # release memory
        ```

        3. Choose an optical flow model and set it to eval mode.

        ```python
        from torchvision.models.optical_flow import raft_large

        model = raft_large(pretrained=True, progress=False).to(device)
        model = model.eval()
        gc.collect() # release memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        ```

        4. Detect flow.

        ```python
        i, frame_gap = 0, 10
        flow_id = i
        flow_img, pred_flow = detect_flow(
                    frames[i], frames[i+frame_gap], 
                    model = model, 
                    device = device, 
                    output_device = "cpu",
                    save_flow_image = True,
                    save_flow_data = True,
                    output_folder = "outputs",
                    flow_image_folder = "flowviz",
                    flow_data_folder = "flowraw",
                    flow_basename = "predicted_flow",
                    flow_id = flow_id, 
                    ext = "png",
                )
        ```

    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = raft_large(pretrained=True, progress=False).to(device)
        model = model.eval()
        _release_memory()

    frameA = _prepare_frame(frameA.to(device), preprocess=preprocess)
    frameB = _prepare_frame(frameB.to(device), preprocess=preprocess)

    list_of_flows = model(frameA, frameB)
    # release memory
    frameA, frameB = None, None
    _release_memory()
    with torch.no_grad():
        predicted_flow = list_of_flows[-1][0].clone()
    # release memory
    list_of_flows = None
    _release_memory()
    with torch.no_grad():
        flow_img = flow_to_image(predicted_flow).to("cpu")
        predicted_flow = predicted_flow.detach().to("cpu")
        # predicted_flow = None
        _release_memory()

    if save_flow_image or save_flow_data:
        if ext.endswith("png"):
            ext = "png"
            write_imfile = write_png
        else:
            ext = "jpg"
            write_imfile = write_jpeg
        if (output_folder is None) or (not output_folder):
            output_folder = "outputs"
        if (flow_basename is None) or (not flow_basename):
            flow_basename = "predicted_flow"
        if (flow_id is not None) and flow_id:
            flow_name = f"{flow_basename}_{flow_id}"
        os.makedirs(output_folder, exist_ok=True)
        
        if save_flow_data:
            flow_data_folder_path = os.path.join(output_folder, flow_data_folder)
            os.makedirs(flow_data_folder_path, exist_ok=True)
            save_raw_path = os.path.join(flow_data_folder_path, f"{flow_name}.pt")
            _, H, W = predicted_flow.shape
            torch.save(predicted_flow, save_raw_path)
        if save_flow_image:
            flow_image_folder_path = os.path.join(output_folder, flow_image_folder)
            os.makedirs(flow_image_folder_path, exist_ok=True)
            save_viz_path = os.path.join(flow_image_folder_path, f"{flow_name}.{ext}")
            write_imfile(flow_img, save_viz_path)

    return flow_img.to(output_device), predicted_flow.to(output_device)


def detect_flows(frames: Tensor, model: nn.Module, 
    frame_gap: int=1, 
    frame_start_index: int=0, 
    quick_test: bool=False, 
    quick_test_steps: int=3, 
    notebook_mode: bool=False,
    return_flow_images: bool=False,
    return_flow_data: bool=False,
    **kwargs) -> Tuple[Optional[List[Tensor]], Optional[List[Tensor]]]:
    """Detects and saves optical flow in a series of frames' tensor of shape ``(N, 3, H, W)``.
    
    Args:
        frames (Tensor): A frames' tensor of shape ``(N, 3, H, W)``
        model (nn.Module, optional): PyTorch model for Optical Flow. 
            (example: ``torchvision.models.optical_flow.raft_large`` model)
        frame_gap (int): The gap between successive frames to evaluate optical flow (O.F.) on.
            (example: a gap of 2 will evaluate O.F. between $i^{th}$ and $(i+2)^{th}$ frames.)
            (default: 1)
        frame_start_index (int): Index of the frame to start detecting O.F. from.
            (default: 0)
        quick_test (bool): If running the function as a quick test or not.
            (default: False)
        quick_test_steps (int): The number of steps to run the ``detect_flow()`` 
            function on, if ``quick_test = True``.
            (default: 3)
        notebook_mode (bool): If running from a jupyter notebook, set this to ``True``.
            (default: False)
        return_flow_images (bool): Whether to return all the flow-images or not.
            (default: True)
        return_flow_data (bool): Whether to return all the flow-data or not.
            (default: True)
        **kwargs: The rest of the keyword argument parameters of ``detect_flow()`` function.

        .. info::
            The following values are set by default for ``detect_flow()`` keyword arguments.

            ```python
            kwargs = {
                "device": None,
                "output_device": "cpu",
                "save_flow_image": True,
                "save_flow_data": True,
                "output_folder": "outputs",
                "flow_image_folder": "flowviz",
                "flow_data_folder": "flowraw",
                "flow_basename": "predicted_flow",
                "ext": "png",
            }
            ```

    Returns:
        flow_imgs (Tensor): Tensor of shape ``(N, 3, H, W)``.
        pred_flows (Tensor): Tensor of shape ``(N, 2, H, W)``.

    Usage:

        1. Download video data.

        ```python
        import gc
        import tempfile
        from pathlib import Path
        from urllib.request import urlretrieve


        video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
        video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
        _ = urlretrieve(video_url, video_path)
        gc.collect() # release memory
        ```

        2. Extract the video frames.

        ```python    
        from torchvision.io import read_video

        frames, _, _ = read_video(str(video_path))
        frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        gc.collect() # release memory
        ```

        3. Choose an optical flow model and set it to eval mode.

        ```python
        from torchvision.models.optical_flow import raft_large

        model = raft_large(pretrained=True, progress=False).to(device)
        model = model.eval()
        gc.collect() # release memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        ```

        4. Detect flow for all the image-pairs. 
        
        .. warning::
            To quickly test if a few 
            image-pairs are working as expected or not, set ``quick_test = True``. 
            Here we instantiate the optical-flow model, and then call the function 
            `detect_flows()`, as shown below. If running from a notebook, set 
            ``notebook_mode = True``. In case, you are running out of memory, 
            consider setting ``return_flow_images = False`` and 
            ``return_flow_data = False``. This will ensure to free-up memory and 
            accordingly return None for ``flow_imgs`` and/or ``pred_flows``.

        ```python
        # Specify keyword-arguments for detect_flow() function.
        kwargs = {
                "device": None,
                "output_device": "cpu",
                "save_flow_image": True,
                "save_flow_data": True,
                "output_folder": "outputs",
                "flow_image_folder": "flowviz",
                "flow_data_folder": "flowraw",
                "flow_basename": "predicted_flow",
                "ext": "png",
            }
        
        # Detect optical flow
        flow_imgs, pred_flows = detect_flows(
            frames, model,
            frame_gap = 1, # detect flow on every successive frame-pairs
            frame_start_index = 0,
            quick_test = False, # Set this to True, to quickly test if this 
                                # works on a few frame-pairs.
            quick_test_steps = 3,
            notebook_mode = False, # Set to True if running from a notebook.
            return_flow_images = True,
            return_flow_data = True,
            **kwargs
        )

        ```
    """
    if not kwargs:
        kwargs = {}

    if notebook_mode:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    flow_imgs = []
    pred_flows = []
    num_frames: int = frames.shape[0]
    total_steps: int = num_frames - frame_gap
    ndigits = len(str(num_frames))
    if quick_test_steps > total_steps:
        quick_test_steps = total_steps
    imax = quick_test_steps if quick_test else total_steps

    # Check keyword agruments for detect_flow() and if necessary, 
    # enforce the following.
    device = kwargs.get("device", None)
    output_device = kwargs.get("output_device", "cpu")
    save_flow_image = kwargs.get("save_flow_image", True)
    save_flow_data = kwargs.get("save_flow_data", True)
    output_folder = kwargs.get("output_folder", "outputs")
    flow_image_folder = kwargs.get("flow_image_folder", "flowviz")
    flow_data_folder = kwargs.get("flow_data_folder", "flowraw")
    flow_basename = kwargs.get("flow_basename", "predicted_flow")
    ext = kwargs.get("ext", "png")

    for i in tqdm(range(imax), desc="Evaluate Flow"):
        flow_id = str(i).zfill(ndigits)
        flow_img, pred_flow = detect_flow(
            frames[i], frames[i+frame_gap], 
            model = model, 
            device = device, 
            output_device = output_device,
            save_flow_image = save_flow_image,
            save_flow_data = save_flow_data,
            output_folder = output_folder,
            flow_image_folder = flow_image_folder,
            flow_data_folder = flow_data_folder,
            flow_basename = flow_basename,
            flow_id = flow_id, 
            ext = "png",
        )
        with torch.no_grad():
            if return_flow_images:
                flow_imgs.append(flow_img.clone())
            else:
                flow_imgs = None

            if return_flow_data:
                pred_flows.append(pred_flow.clone())
            else:
                pred_flows = None
        # release memory
        flow_img = None
        pred_flow = None   
        _release_memory()

    return flow_imgs, pred_flows
