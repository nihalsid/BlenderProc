import torchvision.transforms as T
import torch
import numpy as np
import cv2
from PIL import Image

import vfront_palette as palette
from dutils import dot


def visualize_depth(depth, img_ijs=None, H=None,W=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W) or (N,)
    """
    if H is None and W is None:
        H, W = depth.shape
    if isinstance(depth, torch.Tensor):
        x = depth.cpu().numpy()
    else:
        x = depth
    # convert invalid depth vals to 0
    x[np.isinf(x)] = 0
    x[np.isneginf(x)] = 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    print(mi, ma)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    if img_ijs is not None:
        depth_img = np.zeros((H, W), dtype=np.uint8)
        depth_img[tuple(img_ijs.T)] = x
        x = depth_img
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def visualize_label(label, img_ijs=None, H=None,W=None, palette_name=None):
    """
    label: (H, W) or (N,)
    """
    if H is None and W is None:
        H, W = label.shape
    if palette_name is None:
        from dutils import color_palette
    else:
        color_palette = palette.get(palette_name, format="numpy")

    if isinstance(label, torch.Tensor):
        x = label.cpu().numpy()
    else:
        x =label 
    label_img = np.zeros((H, W, 3), dtype=np.uint8)
    if img_ijs is not None:
        label_img[tuple(img_ijs.T)] = color_palette[x % len(color_palette)]
    else:
        label_img = color_palette[x % len(color_palette)].reshape((H, W, 3))

    return torch.from_numpy(label_img) / 255

def unproject_2d_3d(cam2world, K, d, uv=None, th=None):
    if uv is None and len(d.shape) >= 2:
        # create mesh grid according to d
        uv = np.stack(np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0])), -1)
        uv = uv.reshape(-1, 2)
        d = d.reshape(-1)
        if not isinstance(d, np.ndarray):
            uv = torch.from_numpy(uv).to(d)
        if th is not None:
            uv = uv[d > th]
            d = d[d > th]
    if isinstance(uv, np.ndarray):
        uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)
        cam_point = dot(np.linalg.inv(K), uvh) * d[:, None]
    else:
        uvh = torch.cat([uv, torch.ones(len(uv), 1).to(uv)], -1)
        cam_point = dot(torch.inverse(K), uvh) * d[:, None]

    world_point = dot(cam2world, cam_point)
    return world_point

