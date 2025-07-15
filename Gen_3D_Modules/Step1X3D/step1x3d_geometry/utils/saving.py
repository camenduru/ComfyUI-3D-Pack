import json
import os
import re
import shutil

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import trimesh
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw

from step1x3d_geometry.utils.typing import *


class SaverMixin:
    _save_dir: Optional[str] = None

    def set_save_dir(self, save_dir: str):
        self._save_dir = save_dir

    def get_save_dir(self):
        if self._save_dir is None:
            raise ValueError("Save dir is not set")
        return self._save_dir

    def convert_data(self, data):
        if data is None:
            return None
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list):
            return [self.convert_data(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        else:
            raise TypeError("Data must be numpy.ndarray, torch.Tensor, list or dict, got", type(data))

    def get_save_path(self, filename):
        save_path = os.path.join(self.get_save_dir(), filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

    DEFAULT_RGB_KWARGS = {"data_format": "HWC", "data_range": (0, 1)}
    DEFAULT_UV_KWARGS = {"data_format": "HWC", "data_range": (0, 1), "cmap": "checkerboard"}
    DEFAULT_GRAYSCALE_KWARGS = {"data_range": None, "cmap": "jet"}
    DEFAULT_GRID_KWARGS = {"align": "max"}

    def get_rgb_image_(self, img, data_format, data_range, rgba=False):
        img = self.convert_data(img)
        assert data_format in ["CHW", "HWC"]
        if data_format == "CHW":
            img = img.transpose(1, 2, 0)
        if img.dtype != np.uint8:
            img = img.clip(min=data_range[0], max=data_range[1])
            img = ((img - data_range[0]) / (data_range[1] - data_range[0]) * 255.0).astype(np.uint8)
        nc = 4 if rgba else 3
        imgs = [img[..., start:start + nc] for start in range(0, img.shape[-1], nc)]
        imgs = [img_ if img_.shape[-1] == nc else np.concatenate(
            [img_, np.zeros((img_.shape[0], img_.shape[1], nc - img_.shape[2]), dtype=img_.dtype)], axis=-1)
            for img_ in imgs]
        img = np.concatenate(imgs, axis=1)
        if rgba:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _save_rgb_image(self, filename, img, data_format, data_range):
        img = self.get_rgb_image_(img, data_format, data_range)
        cv2.imwrite(filename, img)

    def save_rgb_image(self, filename, img,
                       data_format=DEFAULT_RGB_KWARGS["data_format"],
                       data_range=DEFAULT_RGB_KWARGS["data_range"]) -> str:
        save_path = self.get_save_path(filename)
        self._save_rgb_image(save_path, img, data_format, data_range)
        return save_path

    def get_uv_image_(self, img, data_format, data_range, cmap):
        img = self.convert_data(img)
        assert data_format in ["CHW", "HWC"]
        if data_format == "CHW":
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in ["checkerboard", "color"]
        if cmap == "checkerboard":
            n_grid = 64
            mask = (img * n_grid).astype(int)
            mask = (mask[..., 0] + mask[..., 1]) % 2 == 0
            img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img[mask] = np.array([255, 0, 255], dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cmap == "color":
            img_ = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_[..., 0] = (img[..., 0] * 255).astype(np.uint8)
            img_[..., 1] = (img[..., 1] * 255).astype(np.uint8)
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            img = img_
        return img

    def save_uv_image(self, filename, img,
                      data_format=DEFAULT_UV_KWARGS["data_format"],
                      data_range=DEFAULT_UV_KWARGS["data_range"],
                      cmap=DEFAULT_UV_KWARGS["cmap"]) -> str:
        save_path = self.get_save_path(filename)
        img = self.get_uv_image_(img, data_format, data_range, cmap)
        cv2.imwrite(save_path, img)
        return save_path

    def get_grayscale_image_(self, img, data_range, cmap):
        img = self.convert_data(img)
        img = np.nan_to_num(img)
        if data_range is None:
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img.clip(data_range[0], data_range[1])
            img = (img - data_range[0]) / (data_range[1] - data_range[0])
        if cmap is None:
            img = (img * 255.0).astype(np.uint8)
            img = np.repeat(img[..., None], 3, axis=2)
        elif cmap == "jet":
            img = (img * 255.0).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        elif cmap == "magma":
            img = 1.0 - img
            base = cm.get_cmap("magma")
            colormap = LinearSegmentedColormap.from_list("magma", base(np.linspace(0, 1, 256)), 256)(np.linspace(0, 1, 256))[:, :3]
            a = np.floor(img * 255.0).astype(np.uint16)
            b = (a + 1).clip(max=255).astype(np.uint16)
            f = img * 255.0 - a
            img = colormap[a] + (colormap[b] - colormap[a]) * f[..., None]
            img = (img * 255.0).astype(np.uint8)
        elif cmap == "spectral":
            colormap = plt.get_cmap("Spectral")
            def blend_rgba(image):
                return image[..., :3] * image[..., -1:] + (1.0 - image[..., -1:])
            img = colormap(img)
            img = blend_rgba(img)
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _save_grayscale_image(self, filename, img, data_range, cmap):
        img = self.get_grayscale_image_(img, data_range, cmap)
        cv2.imwrite(filename, img)

    def save_grayscale_image(self, filename, img,
                             data_range=DEFAULT_GRAYSCALE_KWARGS["data_range"],
                             cmap=DEFAULT_GRAYSCALE_KWARGS["cmap"]) -> str:
        save_path = self.get_save_path(filename)
        self._save_grayscale_image(save_path, img, data_range, cmap)
        return save_path

    def get_image_grid_(self, imgs, align):
        if isinstance(imgs[0], list):
            return np.concatenate([self.get_image_grid_(row, align) for row in imgs], axis=0)
        cols = []
        for col in imgs:
            if col["type"] == "rgb":
                rgb_kwargs = self.DEFAULT_RGB_KWARGS.copy()
                rgb_kwargs.update(col["kwargs"])
                cols.append(self.get_rgb_image_(col["img"], **rgb_kwargs))
            elif col["type"] == "uv":
                uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
                uv_kwargs.update(col["kwargs"])
                cols.append(self.get_uv_image_(col["img"], **uv_kwargs))
            elif col["type"] == "grayscale":
                grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
                grayscale_kwargs.update(col["kwargs"])
                cols.append(self.get_grayscale_image_(col["img"], **grayscale_kwargs))

        if align == "max":
            h, w = max(col.shape[0] for col in cols), max(col.shape[1] for col in cols)
        elif align == "min":
            h, w = min(col.shape[0] for col in cols), min(col.shape[1] for col in cols)
        elif isinstance(align, int):
            h = w = align
        elif isinstance(align, tuple) and isinstance(align[0], int) and isinstance(align[1], int):
            h, w = align
        else:
            raise ValueError(f"Unsupported image grid align: {align}")

        for i in range(len(cols)):
            if cols[i].shape[:2] != (h, w):
                cols[i] = cv2.resize(cols[i], (w, h), interpolation=cv2.INTER_LINEAR)
        return np.concatenate(cols, axis=1)

    def save_image_grid(self, filename, imgs, align=DEFAULT_GRID_KWARGS["align"], texts=None):
        save_path = self.get_save_path(filename)
        img = self.get_image_grid_(imgs, align=align)
        if texts is not None:
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            for i, text in enumerate(texts):
                y = (img.size[1] // len(texts)) * i
                draw.text((1, y), str(text), fill=(0, 0, 0))
            img = np.asarray(img)
        cv2.imwrite(save_path, img)
        return save_path

    def save_image(self, filename, img) -> str:
        save_path = self.get_save_path(filename)
        img = self.convert_data(img)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(save_path, img)
        return save_path

    def save_image_vutils(self, filename, img) -> str:
        save_path = self.get_save_path(filename)
        vutils.save_image(img, save_path)
        return save_path

    def save_cubemap(self, filename, img, data_range=(0, 1), rgba=False) -> str:
        save_path = self.get_save_path(filename)
        img = self.convert_data(img)
        imgs_full = []
        for start in range(0, img.shape[-1], 3):
            img_ = img[..., start:start + 3]
            img_ = np.stack([self.get_rgb_image_(img_[i], "HWC", data_range, rgba=rgba) for i in range(6)], axis=0)
            size = img_.shape[1]
            blank = np.zeros((size, size, 3), dtype=np.uint8)
            img_full = np.concatenate([
                np.concatenate([blank, img_[2], blank, blank], axis=1),
                np.concatenate([img_[1], img_[4], img_[0], img_[5]], axis=1),
                np.concatenate([blank, img_[3], blank, blank], axis=1)
            ], axis=0)
            imgs_full.append(img_full)
        final = np.concatenate(imgs_full, axis=1)
        cv2.imwrite(save_path, final)
        return save_path

    def save_data(self, filename, data) -> str:
        data = self.convert_data(data)
        save_path = self.get_save_path(filename)
        if isinstance(data, dict):
            np.savez(save_path if save_path.endswith('.npz') else save_path + ".npz", **data)
        else:
            np.save(save_path if save_path.endswith('.npy') else save_path + ".npy", data)
        return save_path

    def save_state_dict(self, filename, data) -> str:
        save_path = self.get_save_path(filename)
        torch.save(data, save_path)
        return save_path

    def save_img_sequence(self, filename, img_dir, matcher, save_format="mp4", fps=30) -> str:
        assert save_format in ["gif", "mp4"]
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        save_path = self.get_save_path(filename)
        matcher = re.compile(matcher)
        img_dir = os.path.join(self.get_save_dir(), img_dir)
        files = sorted([f for f in os.listdir(img_dir) if matcher.search(f)],
                       key=lambda f: int(matcher.search(f).groups()[0]))
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(img_dir, f)), cv2.COLOR_BGR2RGB) for f in files]
        imageio.mimsave(save_path, imgs, fps=fps)
        return save_path

    def save_mesh(self, filename, v_pos, t_pos_idx, v_tex=None, t_tex_idx=None) -> str:
        save_path = self.get_save_path(filename)
        mesh = trimesh.Trimesh(vertices=self.convert_data(v_pos), faces=self.convert_data(t_pos_idx))
        mesh.export(save_path)
        return save_path

    def save_file(self, filename, src_path) -> str:
        save_path = self.get_save_path(filename)
        shutil.copyfile(src_path, save_path)
        return save_path

    def save_txt(self, filename, comment) -> str:
        save_path = self.get_save_path(filename)
        with open(save_path, "w") as f:
            f.write(comment)
        return save_path

    def save_json(self, filename, payload) -> str:
        save_path = self.get_save_path(filename)
        with open(save_path, "w") as f:
            f.write(json.dumps(payload))
        return save_path
