"""
OmniObject3D dataset loader for PRISM.

Supports two on-disk layouts (auto-detected; preprocessed is checked first):

**A — Preprocessed** (`blender_renders_24_views/`):

    {data_root}/blender_renders_24_views/
        {category}/{object_id}/{view:04d}/rgb.png, depth.npy, normal.npy, camera.npz
    {data_root}/raw_scans/{category}/{object_id}/{object_id}.obj

**B — Official extracted release** (`blender_renders/`):

    {data_root}/blender_renders/{object_id}/render/
        transforms.json
        images/{file_path}.png
        depths/{file_path}_depth.exr   (opencv-python-headless and/or OpenEXR+Imath)
        normals/{file_path}_normal.png
    {data_root}/raw_scans/{object_id}/Scan/Scan.obj

The dataset returns one (object, view) per sample; a random view is chosen in __getitem__.
"""

import json
import hashlib
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_INDEX_CACHE_VERSION = 1
_INDEX_MEMO: Dict[str, Dict[str, Any]] = {}


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "prism" / "omniobject3d"


def _root_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]


def _quick_signature(root: Path) -> str:
    """
    Cheap invalidation signature from immediate children metadata.
    Avoids expensive deep hashing while catching common dataset updates.
    """
    parts: List[str] = [str(root.resolve())]
    try:
        st_root = root.stat()
        parts.append(f"root:{st_root.st_mtime_ns}:{st_root.st_size}")
    except OSError:
        parts.append("root:missing")
    try:
        for de in sorted(os.scandir(root), key=lambda x: x.name):
            if not de.is_dir(follow_symlinks=False):
                continue
            try:
                st = de.stat(follow_symlinks=False)
                parts.append(f"{de.name}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                parts.append(f"{de.name}:err")
    except OSError:
        parts.append("scan:err")
    s = "|".join(parts).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def _cache_file(layout: str, render_root: Path) -> Path:
    return _cache_dir() / f"{layout}-{_root_id(render_root.resolve())}.json"


def _serialize_entry(layout: str, payload: Dict[str, Any], signature: str, render_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "version": _INDEX_CACHE_VERSION,
        "layout": layout,
        "render_root": str(render_root.resolve()),
        "signature": signature,
        "objects": [[c, o] for (c, o) in payload["objects"]],
    }
    if layout == "extracted":
        meta_out: Dict[str, Any] = {}
        for (cat, oid), m in payload["meta"].items():
            meta_out[f"{cat}|{oid}"] = {
                "render_dir": str(m["render_dir"]),
                "frames": m["frames"],
                "camera_angle_x": float(m["camera_angle_x"]),
                "wh": [int(m["wh"][0]), int(m["wh"][1])],
            }
        out["meta"] = meta_out
    return out


def _deserialize_entry(layout: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "objects": [(str(c), str(o)) for c, o in raw.get("objects", [])],
        "meta": {},
    }
    if layout == "extracted":
        meta_raw = raw.get("meta", {})
        meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for k, m in meta_raw.items():
            if "|" not in k:
                continue
            cat, oid = k.split("|", 1)
            meta[(cat, oid)] = {
                "render_dir": Path(m["render_dir"]),
                "frames": m["frames"],
                "camera_angle_x": float(m["camera_angle_x"]),
                "wh": (int(m["wh"][0]), int(m["wh"][1])),
            }
        payload["meta"] = meta
    return payload


# ---------------------------------------------------------------------------
# Path helpers — preprocessed layout (per-view folders 0000 …)
# ---------------------------------------------------------------------------

def _rgb_path(obj_dir: Path, view_id: int) -> Path:
    return obj_dir / f"{view_id:04d}" / "rgb.png"


def _depth_path(obj_dir: Path, view_id: int) -> Path:
    return obj_dir / f"{view_id:04d}" / "depth.npy"


def _normal_path(obj_dir: Path, view_id: int) -> Path:
    return obj_dir / f"{view_id:04d}" / "normal.npy"


def _camera_path(obj_dir: Path, view_id: int) -> Path:
    return obj_dir / f"{view_id:04d}" / "camera.npz"


def _infer_category(obj_id: str) -> str:
    if "_" in obj_id:
        return obj_id.split("_", 1)[0]
    return "object"


def _intrinsics_from_angle_x(camera_angle_x: float, w: int, h: int) -> np.ndarray:
    """Pinhole K from horizontal FOV (radians); vertical FOV from aspect ratio."""
    fx = 0.5 * float(w) / np.tan(0.5 * camera_angle_x)
    angle_y = 2.0 * np.arctan(np.tan(0.5 * camera_angle_x) * float(h) / float(w))
    fy = 0.5 * float(h) / np.tan(0.5 * angle_y)
    cx = 0.5 * float(w)
    cy = 0.5 * float(h)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_exr_depth(path: Path) -> np.ndarray:
    """
    Load float depth from Blender-style EXR.

    Order: **PyOpenEXR** first (avoids OpenCV printing the same EXR-disabled warning
    on every ``imread``). **OpenCV** only as fallback, with log level lowered for
    that call (many wheels ship without EXR; see ``OPENCV_IO_ENABLE_OPENEXR``).
    """
    path = Path(path)

    # 1) PyOpenEXR (``pip install OpenEXR`` — includes Imath bindings).
    try:
        import Imath
        import OpenEXR
    except ImportError:
        pass
    else:
        exr = OpenEXR.InputFile(str(path))
        try:
            header = exr.header()
            dw = header["dataWindow"]
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            chmap = header["channels"]
            names = list(chmap.keys())
            preferred = ["Z", "Y", "V", "R", "G", "B"]
            ordered = [n for n in preferred if n in chmap] + [n for n in names if n not in preferred]

            float_pt = Imath.PixelType(Imath.PixelType.FLOAT)
            half_pt = Imath.PixelType(Imath.PixelType.HALF)
            for name in ordered:
                for pt, np_dtype in ((float_pt, np.float32), (half_pt, np.float16)):
                    try:
                        raw = exr.channel(name, pt)
                    except Exception:
                        continue
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape((h, w))
                    return arr.astype(np.float32, copy=False)
            raise ValueError(f"No FLOAT/HALF channel could be read from EXR: {path}")
        finally:
            exr.close()

    # 2) OpenCV fallback (suppress per-read WARN spam from grfmt_exr.cpp).
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "Reading .exr depth maps: install one of:\n"
            "  • pip install OpenEXR                  (recommended), or\n"
            "  • pip install opencv-python-headless (EXR must be enabled in build).\n"
            "Do not `pip install Imath` alone — PyPI's unrelated `Imath` package conflicts.\n"
            "Building OpenEXR from source may need system libOpenEXR / cmake."
        ) from e

    _log = getattr(cv2, "utils", None)
    _prev = None
    if _log is not None and hasattr(_log, "logging"):
        try:
            _prev = _log.logging.getLogLevel()
            _log.logging.setLogLevel(_log.logging.LOG_LEVEL_ERROR)
        except Exception:
            _prev = None
    try:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    finally:
        if _prev is not None:
            try:
                _log.logging.setLogLevel(_prev)
            except Exception:
                pass

    if arr is not None and arr.size > 0:
        if arr.ndim == 3:
            arr = arr[..., 0]
        return np.asarray(arr, dtype=np.float32)

    raise ImportError(
        "Reading .exr depth maps: ``pip install OpenEXR`` failed to import earlier, "
        "and OpenCV could not decode this EXR (codec often disabled in pip wheels). "
        "Install OpenEXR or use a build with ``OPENCV_IO_ENABLE_OPENEXR=1``."
    )


def _load_mask(pil_img: "Image.Image") -> np.ndarray:
    """
    (H, W) float32 foreground mask.
    Background = dark pixels that are connected to the image border (flood-fill).
    Interior dark spots (eyes, markings, transparent holes) are kept as foreground.
    """
    if pil_img.mode in ("RGBA", "LA"):
        dark = np.array(pil_img)[:, :, -1] <= 10
    else:
        rgb = np.array(pil_img.convert("RGB")).astype(np.float32)
        dark = rgb.sum(-1) <= 10.0

    try:
        from scipy.ndimage import label as nd_label
        labeled, _ = nd_label(dark)
        border_lbls: set = set()
        for arr in (labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1]):
            border_lbls.update(arr.tolist())
        border_lbls.discard(0)
        bg = np.zeros(dark.shape, dtype=bool)
        for lbl in border_lbls:
            bg |= labeled == lbl
        return (~bg).astype(np.float32)
    except ImportError:
        return (~dark).astype(np.float32)


def _scale_K(K: np.ndarray, orig_wh: Tuple[int, int], new_wh: Tuple[int, int]) -> np.ndarray:
    ow, oh = orig_wh
    nw, nh = new_wh
    sx, sy = nw / ow, nh / oh
    Ks = K.copy()
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] *= sx
    Ks[1, 2] *= sy
    return Ks.astype(np.float32)


def _build_index(layout: str, render_root: Path) -> Dict[str, Any]:
    if layout == "preprocessed":
        objects: List[Tuple[str, str]] = []
        for cat_dir in sorted(render_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            for obj_dir in sorted(cat_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue
                if _rgb_path(obj_dir, 0).exists():
                    objects.append((cat_dir.name, obj_dir.name))
        return {"objects": objects, "meta": {}}

    # extracted layout
    objects = []
    meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for oid_dir in sorted(render_root.iterdir()):
        if not oid_dir.is_dir():
            continue
        obj_id = oid_dir.name
        cat = _infer_category(obj_id)
        render_dir = oid_dir / "render"
        tpath = render_dir / "transforms.json"
        if not tpath.exists():
            continue
        with open(tpath) as f:
            meta_json = json.load(f)
        frames: List = meta_json.get("frames") or []
        if not frames:
            continue
        fp0 = frames[0]["file_path"]
        rgb0 = render_dir / "images" / f"{fp0}.png"
        if not rgb0.exists():
            continue
        with Image.open(rgb0) as im:
            w, h = im.size
        cam_ax = float(meta_json["camera_angle_x"])
        key = (cat, obj_id)
        meta[key] = {
            "render_dir": render_dir,
            "frames": frames,
            "camera_angle_x": cam_ax,
            "wh": (w, h),
        }
        objects.append((cat, obj_id))
    return {"objects": objects, "meta": meta}


def _get_index_cached(layout: str, render_root: Path) -> Dict[str, Any]:
    signature = _quick_signature(render_root)
    memo_key = f"{layout}|{render_root.resolve()}|{signature}"
    if memo_key in _INDEX_MEMO:
        cached = _INDEX_MEMO[memo_key]
        return {"objects": list(cached["objects"]), "meta": dict(cached["meta"])}

    cfile = _cache_file(layout, render_root)
    payload: Optional[Dict[str, Any]] = None
    try:
        if cfile.exists():
            with open(cfile) as f:
                raw = json.load(f)
            if (
                raw.get("version") == _INDEX_CACHE_VERSION
                and raw.get("layout") == layout
                and raw.get("signature") == signature
                and raw.get("render_root") == str(render_root.resolve())
            ):
                payload = _deserialize_entry(layout, raw)
    except Exception:
        payload = None

    if payload is None:
        payload = _build_index(layout, render_root)
        try:
            cdir = _cache_dir()
            cdir.mkdir(parents=True, exist_ok=True)
            with open(cfile, "w") as f:
                json.dump(_serialize_entry(layout, payload, signature, render_root), f)
        except Exception:
            pass

    _INDEX_MEMO[memo_key] = payload
    return {"objects": list(payload["objects"]), "meta": dict(payload["meta"])}


def enumerate_object_pairs(
    data_root: str | Path,
    categories: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    All ``(category, object_id)`` pairs on disk in the same stable order as
    ``OmniObject3DDataset`` uses before train/val/test splitting.
    """
    data_root = Path(data_root)
    pre_root = data_root / "blender_renders_24_views"
    ext_root = data_root / "blender_renders"
    if pre_root.is_dir():
        objects = _get_index_cached("preprocessed", pre_root)["objects"]
        if categories:
            cset = set(categories)
            objects = [o for o in objects if o[0] in cset]
        return objects
    if ext_root.is_dir():
        objects = _get_index_cached("extracted", ext_root)["objects"]
        if categories:
            cset = set(categories)
            objects = [o for o in objects if o[0] in cset]
        return objects
    raise FileNotFoundError(
        f"OmniObject3D data not found under {data_root}: "
        f"expected {pre_root} or {ext_root}."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OmniObject3DDataset(Dataset):
    """
    Args:
        data_root:    Path to OmniObject3D root (see module docstring).
        split:        'train' | 'val' | 'test'
        n_views:      Max views per object (24 for preprocessed; capped by frame count for extracted).
        image_size:   Square resize; default 800 (no resize when native size is 800).
        split_file:   Optional JSON {split: [object_ids]}.
        categories:   Optional list of category names (preprocessed: folder name; extracted: prefix before '_').
        restrict_object_ids: If set, only these ``object_id`` strings (e.g. ``clock_029``) are kept,
            in the given order; train/val/test splits and ``split_file`` are ignored for membership.
        virtual_epoch_len: If set, ``__len__`` returns this value and indices wrap so each index
            draws a **new** random multi-view sample for the same object list (for single-object overfit).
    """

    N_VIEWS = 24

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        n_views: int = 24,
        image_size: int = 800,
        split_file: Optional[str] = None,
        categories: Optional[List[str]] = None,
        n_input_views: int = 1,
        restrict_object_ids: Optional[Sequence[str]] = None,
        virtual_epoch_len: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.scan_root = self.data_root / "raw_scans"
        self.split = split
        self.n_views = n_views
        self.image_size = image_size
        self.n_input_views = n_input_views

        pre_root = self.data_root / "blender_renders_24_views"
        ext_root = self.data_root / "blender_renders"
        if pre_root.is_dir():
            self._layout = "preprocessed"
            self.render_root = pre_root
        elif ext_root.is_dir():
            self._layout = "extracted"
            self.render_root = ext_root
        else:
            raise FileNotFoundError(
                f"OmniObject3D data not found under {self.data_root}: "
                f"expected {pre_root} or {ext_root}."
            )

        self._extracted_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self.to_tensor = transforms.ToTensor()
        self.img_resize = (
            transforms.Resize((image_size, image_size), antialias=True)
            if image_size != 800
            else None
        )

        if self._layout == "preprocessed":
            all_objects = self._enumerate_preprocessed(categories)
        else:
            all_objects = self._enumerate_extracted(categories)

        if restrict_object_ids is not None:
            wanted = list(dict.fromkeys(restrict_object_ids))
            wanted_set = set(wanted)
            self.objects = [o for o in all_objects if o[1] in wanted_set]
            if len(self.objects) == 0:
                raise RuntimeError(
                    f"No objects match restrict_object_ids={list(wanted)!r} under {self.render_root} "
                    f"(layout={self._layout})."
                )
            rank = {pid: i for i, pid in enumerate(wanted)}
            self.objects.sort(key=lambda o: (rank[o[1]], o[0], o[1]))
            if self._layout == "extracted":
                keep = set(self.objects)
                self._extracted_meta = {k: v for k, v in self._extracted_meta.items() if k in keep}
        elif split_file is not None:
            with open(split_file) as f:
                split_map = json.load(f)
            ids_in_split = set(split_map[split])
            self.objects = [o for o in all_objects if o[1] in ids_in_split]
        else:
            self.objects = self._auto_split(all_objects, split)

        if len(self.objects) == 0:
            raise RuntimeError(
                f"No objects found for split='{split}' under {self.render_root} "
                f"(layout={self._layout})."
            )

        self._virtual_epoch_len = virtual_epoch_len
        if virtual_epoch_len is not None and virtual_epoch_len < 1:
            raise ValueError("virtual_epoch_len must be >= 1 when set")

    def _enumerate_preprocessed(self, categories: Optional[List[str]]) -> List[Tuple[str, str]]:
        payload = _get_index_cached("preprocessed", self.render_root)
        objects: List[Tuple[str, str]] = payload["objects"]
        if categories:
            cset = set(categories)
            objects = [o for o in objects if o[0] in cset]
        return objects

    def _enumerate_extracted(self, categories: Optional[List[str]]) -> List[Tuple[str, str]]:
        payload = _get_index_cached("extracted", self.render_root)
        objects = payload["objects"]
        meta = payload["meta"]
        if categories:
            cset = set(categories)
            objects = [o for o in objects if o[0] in cset]
        keep = set(objects)
        self._extracted_meta = {k: v for k, v in meta.items() if k in keep}
        return objects

    def _auto_split(self, objects: list, split: str) -> list:
        rng = random.Random(42)
        shuffled = objects[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(0.90 * n)
        n_val = int(0.05 * n)
        if split == "train":
            return shuffled[:n_train]
        if split == "val":
            return shuffled[n_train : n_train + n_val]
        return shuffled[n_train + n_val :]

    def __len__(self) -> int:
        if self._virtual_epoch_len is not None:
            return self._virtual_epoch_len
        return len(self.objects)

    def __getitem__(self, idx: int) -> dict:
        cat, obj_id = self.objects[idx % len(self.objects)]

        if self._layout == "preprocessed":
            return self._getitem_preprocessed(cat, obj_id)
        return self._getitem_extracted(cat, obj_id)

    def _getitem_preprocessed(self, cat: str, obj_id: str) -> dict:
        obj_dir = self.render_root / cat / obj_id
        n = self.n_input_views
        n_avail = self.n_views

        # Sample n unique context views; pad with random repeats if n > n_avail.
        view_ids = random.sample(range(n_avail), min(n, n_avail))
        while len(view_ids) < n:
            view_ids.append(random.randint(0, n_avail - 1))

        # Build stacked image tensor; collect cameras; extract mask from view 0 (target).
        img_list:  List[torch.Tensor] = []
        c2w_list:  List[torch.Tensor] = []
        K_list:    List[torch.Tensor] = []
        mask_np:   Optional[np.ndarray] = None
        for i, vid in enumerate(view_ids):
            pil_raw = Image.open(_rgb_path(obj_dir, vid))
            if i == 0:
                mask_np = _load_mask(pil_raw)
            rgb_img = pil_raw.convert("RGB")
            if self.img_resize:
                rgb_img = self.img_resize(rgb_img)
            img_list.append(self.to_tensor(rgb_img))
            cam = np.load(_camera_path(obj_dir, vid))
            c2w_list.append(torch.from_numpy(cam["c2w"].astype(np.float32)))
            K_list.append(torch.from_numpy(cam["K"].astype(np.float32)))

        images      = torch.stack(img_list, dim=0)   # (N, 3, H, W)
        input_c2ws  = torch.stack(c2w_list, dim=0)   # (N, 4, 4)
        input_Ks    = torch.stack(K_list,   dim=0)   # (N, 3, 3)
        target_view = view_ids[0]
        c2w = c2w_list[0]
        K   = K_list[0]

        mask = torch.from_numpy(mask_np).unsqueeze(0)   # (1, H_orig, W_orig)
        if self.img_resize:
            mask = F.interpolate(mask.unsqueeze(0),
                                 (self.image_size, self.image_size),
                                 mode="nearest").squeeze(0)

        depth_np = np.load(_depth_path(obj_dir, target_view)).astype(np.float32)
        depth = torch.from_numpy(depth_np).unsqueeze(0)

        normal_np = np.load(_normal_path(obj_dir, target_view)).astype(np.float32)
        normal = torch.from_numpy(normal_np).permute(2, 0, 1)
        normal = F.normalize(normal, dim=0)

        mesh_path = str(self.scan_root / cat / obj_id / f"{obj_id}.obj")

        return {
            "images":     images,
            "input_c2ws": input_c2ws,
            "input_Ks":   input_Ks,
            "depth":      depth,
            "normal":     normal,
            "mask":       mask,
            "c2w":        c2w,
            "K":          K,
            "category":   cat,
            "object_id":  obj_id,
            "view_id":    target_view,
            "mesh_path":  mesh_path,
        }

    def _getitem_extracted(self, cat: str, obj_id: str) -> dict:
        meta = self._extracted_meta[(cat, obj_id)]
        frames: List = meta["frames"]
        render_dir: Path = meta["render_dir"]
        w0, h0 = meta["wh"]
        cam_ax = meta["camera_angle_x"]

        n_avail = min(self.n_views, len(frames))
        n = self.n_input_views

        # Sample n unique context views; pad with random repeats if n > n_avail.
        view_ids = random.sample(range(n_avail), min(n, n_avail))
        while len(view_ids) < n:
            view_ids.append(random.randint(0, n_avail - 1))

        # Build stacked image tensor; collect cameras; extract mask and orig_wh from view 0.
        img_list:  List[torch.Tensor] = []
        c2w_list:  List[torch.Tensor] = []
        K_list:    List[torch.Tensor] = []
        mask_np:   Optional[np.ndarray] = None
        orig_wh:   Optional[Tuple[int, int]] = None
        K_np_base = _intrinsics_from_angle_x(cam_ax, w0, h0)
        if self.img_resize:
            K_np_scaled = _scale_K(K_np_base, (w0, h0), (self.image_size, self.image_size))
        else:
            K_np_scaled = K_np_base

        for i, vid in enumerate(view_ids):
            fp = frames[vid]["file_path"]
            pil_raw = Image.open(render_dir / "images" / f"{fp}.png")
            if i == 0:
                mask_np = _load_mask(pil_raw)
                orig_wh = (pil_raw.width, pil_raw.height)
            rgb_img = pil_raw.convert("RGB")
            if self.img_resize:
                rgb_img = self.img_resize(rgb_img)
            img_list.append(self.to_tensor(rgb_img))
            c2w_list.append(torch.from_numpy(
                np.array(frames[vid]["transform_matrix"], dtype=np.float32)
            ))
            K_list.append(torch.from_numpy(K_np_scaled))

        images      = torch.stack(img_list, dim=0)   # (N, 3, H, W)
        input_c2ws  = torch.stack(c2w_list, dim=0)   # (N, 4, 4)
        input_Ks    = torch.stack(K_list,   dim=0)   # (N, 3, 3)
        target_view = view_ids[0]
        c2w = c2w_list[0]
        K   = K_list[0]

        target_fp = frames[target_view]["file_path"]

        mask = torch.from_numpy(mask_np).unsqueeze(0)   # (1, H_orig, W_orig)
        if self.img_resize:
            mask = F.interpolate(mask.unsqueeze(0),
                                 (self.image_size, self.image_size),
                                 mode="nearest").squeeze(0)

        depth_np = _load_exr_depth(render_dir / "depths" / f"{target_fp}_depth.exr").astype(np.float32)
        depth = torch.from_numpy(depth_np).unsqueeze(0)
        if self.img_resize:
            depth = F.interpolate(
                depth.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)

        nimg = Image.open(render_dir / "normals" / f"{target_fp}_normal.png").convert("RGB")
        if self.img_resize:
            nimg = self.img_resize(nimg)
        narr = np.asarray(nimg).astype(np.float32) / 255.0
        normal_np = (narr * 2.0) - 1.0
        normal = torch.from_numpy(normal_np).permute(2, 0, 1)
        normal = F.normalize(normal, dim=0)

        mesh_path = str(self.scan_root / obj_id / "Scan" / "Scan.obj")

        return {
            "images":     images,
            "input_c2ws": input_c2ws,
            "input_Ks":   input_Ks,
            "depth":      depth,
            "normal":     normal,
            "mask":       mask,
            "c2w":        c2w,
            "K":          K,
            "category":   cat,
            "object_id":  obj_id,
            "view_id":    target_view,
            "mesh_path":  mesh_path,
        }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 800,
    split_file: Optional[str] = None,
    categories: Optional[List[str]] = None,
    pin_memory: bool = True,
    n_input_views: int = 1,
) -> dict[str, DataLoader]:
    datasets = {
        split: OmniObject3DDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            split_file=split_file,
            categories=categories,
            n_input_views=n_input_views,
        )
        for split in ("train", "val")
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        ),
    }
    return loaders
