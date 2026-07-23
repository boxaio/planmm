import ast
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mesh import extract_submesh, read_obj


_DATASET_DIR = Path(__file__).resolve().parent

NoseRegion = Literal['nose_0', 'nose_1']

_DEFAULT_REGION_FID_TXT: dict[NoseRegion, Path] = {
    'nose_0': _DATASET_DIR / 'bfm_nose_0_fids.txt',
    'nose_1': _DATASET_DIR / 'bfm_nose_1_fids.txt',
}

CREMAD_root = '/media/ubuntu/SSD/PHACK_data/bfm_meshes/cremad'
MEAD_root = '/media/ubuntu/SSD/PHACK_data/bfm_meshes/mead'

TRAIN_BFM_NOSE_PT = _DATASET_DIR / 'train_bfm_nose.pt'
VAL_BFM_NOSE_PT = _DATASET_DIR / 'val_bfm_nose.pt'
TEST_BFM_NOSE_PT = _DATASET_DIR / 'test_bfm_nose.pt'


def _load_seg_face_indices_txt(txt_path: Union[str, Path]) -> list[int]:
    p = Path(txt_path)
    raw = p.read_text(encoding='utf-8').strip()
    if raw.startswith('['):
        seg_fids = ast.literal_eval(raw)
        if not isinstance(seg_fids, list):
            raise ValueError(f'{p}: expected list literal, got {type(seg_fids).__name__}')
        return [int(x) for x in seg_fids]
    out: list[int] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            out.append(int(line))
    return out


def _combine_region_face_indices(names: Sequence[NoseRegion]) -> np.ndarray:
    if not names:
        raise ValueError('regions 非空时才能合并面索引')
    merged: list[int] = []
    for name in dict.fromkeys(names):
        if name not in _DEFAULT_REGION_FID_TXT:
            raise ValueError(f'未知 region: {name!r}，可选: {tuple(_DEFAULT_REGION_FID_TXT)}')
        merged.extend(_load_seg_face_indices_txt(_DEFAULT_REGION_FID_TXT[name]))
    return np.asarray(sorted(set(merged)), dtype=np.int64)


def _submesh_vertex_pick_and_faces(
    faces_full: np.ndarray, region_face_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    vmax = int(faces_full.max()) if faces_full.size else -1
    n_dummy = vmax + 1
    if n_dummy < 0:
        n_dummy = 0
    dummy_v = np.zeros((n_dummy, 3), dtype=np.float32)
    _, sub_faces, vmap = extract_submesh(
        dummy_v, faces_full, region_face_indices, mode='face'
    )
    vtx_pick = np.array(sorted(vmap.keys(), key=lambda k: vmap[k]), dtype=np.int64)
    return vtx_pick, sub_faces


def _resolve_pose_npz_for_target_stem(target_stem: str, target_root: Path) -> Path:
    """在 ``target_root`` 下选取文件名（含扩展名）中含 ``pose`` 的 ``.npz``。

    - 若仅有一个候选，直接返回（不要求文件名含 target stem）。
    - 若有多个，则用 ``target_stem`` 出现在文件名/stem 中的唯一匹配；否则报错提示歧义。
    """
    cands = sorted(
        p for p in target_root.glob('*.npz') if 'pose' in p.name.lower()
    )
    if not cands:
        raise FileNotFoundError(
            f'未在 {target_root} 下找到文件名含 pose 的 .npz'
        )
    if len(cands) == 1:
        return cands[0]
    
    pose_npz = target_stem.replace('_mid_mesh', '_pose') + '.npz'
    pose_npz = os.path.join(target_root, pose_npz)
    assert os.path.exists(pose_npz)
    return pose_npz

def _load_target_pose_from_npz(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path)
    if 'trans' not in z.files:
        raise KeyError(f'{npz_path}: npz 缺少数组 trans')
    if 'angle' not in z.files:
        raise KeyError(f'{npz_path}: npz 缺少数组 angle')
    trans = np.asarray(z['trans'], dtype=np.float32).reshape(-1)
    angle = np.asarray(z['angle'], dtype=np.float32).reshape(-1)
    return np.concatenate([trans, angle], axis=0)


def _pair_paths_from_roots(
    source_root: Path,
    target_root: Path,
    ext: str,
    seed: Optional[int],
) -> list[Tuple[Path, Path]]:
    src_list = sorted(source_root.glob(f'*{ext}'), key=lambda p: p.as_posix())
    tgt_list = sorted(target_root.glob(f'*{ext}'), key=lambda p: p.as_posix())
    if not src_list or not tgt_list:
        raise ValueError(
            f'Need at least one *{ext} in each root; got '
            f'{len(src_list)} under {source_root}, {len(tgt_list)} under {target_root}'
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(src_list)
    rng.shuffle(tgt_list)
    n = min(len(src_list), len(tgt_list))
    return list(zip(src_list[:n], tgt_list[:n]))


def _load_pair_for_cache(
    source_path: Path,
    target_path: Path,
    faces_full: np.ndarray,
    vtx_pick: Optional[np.ndarray],
    target_root: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """读一对 OBJ；面表与 ``faces_full`` 对齐后对顶点做子网格选取（``vtx_pick`` 非空时）。

    若 ``target_root`` 非空，在同目录下按 stem 解析含 ``pose`` 的 ``.npz``，将 ``trans`` 与 ``angle``
    展平后拼接为 ``target_pose``（numpy 1D）。
    """
    obj_s = read_obj(str(source_path), tri=True)
    obj_t = read_obj(str(target_path), tri=True)
    source_verts = np.asarray(obj_s.vs, dtype=np.float32)
    target_verts = np.asarray(obj_t.vs, dtype=np.float32)
    faces = np.asarray(obj_s.fvs, dtype=np.int64)
    faces_t = np.asarray(obj_t.fvs, dtype=np.int64)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f'{source_path}: expected triangle faces (F, 3), got shape {faces.shape}'
        )
    if source_verts.shape[0] != target_verts.shape[0]:
        raise ValueError(
            f'{source_path.stem}: vertex count mismatch '
            f'source {source_verts.shape[0]} vs target {target_verts.shape[0]}'
        )
    if faces.shape != faces_full.shape or not np.array_equal(faces, faces_full):
        raise ValueError(
            f'{source_path.stem}: source faces differ from reference topology'
        )
    if not np.array_equal(faces_t, faces_full):
        raise ValueError(
            f'{target_path.stem}: target faces differ from reference topology'
        )

    vmax = int(faces.max()) if faces.size else -1
    if vmax >= source_verts.shape[0]:
        raise ValueError(
            f'{source_path}: face index max {vmax} >= num verts {source_verts.shape[0]}'
        )

    if vtx_pick is not None:
        source_verts = source_verts[vtx_pick]
        target_verts = target_verts[vtx_pick]

    stem = f'{source_path.stem}__{target_path.stem}'

    if target_root is not None:
        pose_npz = _resolve_pose_npz_for_target_stem(target_path.stem, target_root)
        target_pose = _load_target_pose_from_npz(pose_npz)
    else:
        target_pose = np.array([], dtype=np.float32)

    return source_verts, target_verts, stem, target_pose


_PACK_FACES_FULL: Optional[np.ndarray] = None
_PACK_VTX_PICK: Optional[np.ndarray] = None
_PACK_TGT_ROOT: Optional[Path] = None


def _pack_worker_init(
    faces_full: np.ndarray,
    vtx_pick: Optional[np.ndarray],
    target_root: Optional[Path],
) -> None:
    global _PACK_FACES_FULL, _PACK_VTX_PICK, _PACK_TGT_ROOT
    _PACK_FACES_FULL = faces_full
    _PACK_VTX_PICK = None if vtx_pick is None else vtx_pick.copy()
    _PACK_TGT_ROOT = None if target_root is None else Path(target_root)


def _pack_worker_one(pair_paths: Tuple[str, str]) -> Dict:
    assert _PACK_FACES_FULL is not None
    src_p, tgt_p = Path(pair_paths[0]), Path(pair_paths[1])
    sv, tv, stem, pose = _load_pair_for_cache(
        src_p, tgt_p, _PACK_FACES_FULL, _PACK_VTX_PICK, _PACK_TGT_ROOT
    )
    out: Dict = {
        'source_verts': torch.from_numpy(np.ascontiguousarray(sv)),
        'target_verts': torch.from_numpy(np.ascontiguousarray(tv)),
        'stem': stem,
        'target_stem': tgt_p.stem,
    }
    if pose.size > 0:
        out['target_pose'] = torch.from_numpy(np.ascontiguousarray(pose))
    else:
        out['target_pose'] = None
    return out


def _split_811(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError('split 需要至少 1 对样本')
    perm = rng.permutation(n)
    n_train = n * 8 // 10
    n_val = n * 1 // 10
    n_test = n - n_train - n_val
    if n_train == 0:
        n_train, n_val = 1, 0
        n_test = n - 1
    i1 = n_train
    i2 = i1 + n_val
    return perm[:i1], perm[i1:i2], perm[i2:]


def _pack_split_items(
    pairs: Sequence[Tuple[Path, Path]],
    indices: np.ndarray,
    region_face_indices: Optional[np.ndarray],
    num_workers: int = 4,
    target_root: Optional[Path] = None,
    progress_desc: str = 'pack',
) -> dict:
    ref_src, _ = pairs[0]
    obj0 = read_obj(str(ref_src), tri=True)
    faces_full = np.asarray(obj0.fvs, dtype=np.int64)
    if faces_full.ndim != 2 or faces_full.shape[1] != 3:
        raise ValueError(
            f'{ref_src}: expected triangle faces (F, 3), got shape {faces_full.shape}'
        )
    if region_face_indices is not None:
        vtx_pick, faces_ref = _submesh_vertex_pick_and_faces(
            faces_full, region_face_indices
        )
    else:
        vtx_pick = None
        faces_ref = faces_full

    task_paths: list[Tuple[str, str]] = [
        (str(pairs[int(j)][0]), str(pairs[int(j)][1])) for j in indices
    ]

    if num_workers > 0 and len(task_paths) > 1:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_pack_worker_init,
            initargs=(faces_full, vtx_pick, target_root),
        ) as ex:
            future_to_i = {
                ex.submit(_pack_worker_one, tp): i
                for i, tp in enumerate(task_paths)
            }
            items_batch: list[Optional[dict]] = [None] * len(task_paths)
            for fut in tqdm(
                as_completed(future_to_i),
                total=len(task_paths),
                desc=progress_desc,
                mininterval=0.0,
                smoothing=0,
                dynamic_ncols=True,
                file=sys.stdout,
            ):
                idx = future_to_i[fut]
                items_batch[idx] = fut.result()
            assert all(x is not None for x in items_batch)
            items = items_batch  # type: ignore[assignment]
    else:
        items = []
        for pp in tqdm(
            task_paths,
            desc=progress_desc,
            mininterval=0.0,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            sv, tv, stem, pose = _load_pair_for_cache(
                Path(pp[0]), Path(pp[1]), faces_full, vtx_pick, target_root
            )
            it: Dict = {
                'source_verts': torch.from_numpy(np.ascontiguousarray(sv)),
                'target_verts': torch.from_numpy(np.ascontiguousarray(tv)),
                'stem': stem,
                'target_stem': Path(pp[1]).stem,
            }
            if pose.size > 0:
                it['target_pose'] = torch.from_numpy(np.ascontiguousarray(pose))
            else:
                it['target_pose'] = None
            items.append(it)

    return {
        'faces': torch.from_numpy(np.ascontiguousarray(faces_ref)),
        'items': items,
    }


def build_bfm_nose_pt_cache(
    source_root: Union[str, Path] = CREMAD_root,
    target_root: Union[str, Path] = MEAD_root,
    regions: Sequence[NoseRegion] = ('nose_0', 'nose_1'),
    pairing_seed: Optional[int] = None,
    split_seed: int = 0,
    out_dir: Union[str, Path] = _DATASET_DIR,
    extension: str = '.obj',
    train_path: Union[str, Path] = TRAIN_BFM_NOSE_PT,
    val_path: Union[str, Path] = VAL_BFM_NOSE_PT,
    test_path: Union[str, Path] = TEST_BFM_NOSE_PT,
    num_workers: int = 4,
) -> None:
    """
    从两侧目录做随机配对（与 BFMDataset 根路径模式一致），再按 8:1:1 划分 train/val/test，
    预处理网格并保存 ``train_bfm_nose.pt`` / ``val_bfm_nose.pt`` / ``test_bfm_nose.pt``。

    ``num_workers`` > 0 时用多进程读 OBJ（适合样本多时）；0 表示单进程。
    """
    src_r = Path(source_root)
    tgt_r = Path(target_root)
    out_d = Path(out_dir)
    if not src_r.is_dir():
        raise FileNotFoundError(f'source_root not found: {src_r}')
    if not tgt_r.is_dir():
        raise FileNotFoundError(f'target_root not found: {tgt_r}')
    out_d.mkdir(parents=True, exist_ok=True)

    ext = extension if extension.startswith('.') else f'.{extension}'
    pairs = _pair_paths_from_roots(src_r, tgt_r, ext, pairing_seed)
    region_idx = _combine_region_face_indices(regions)

    n = len(pairs)
    rng_split = np.random.default_rng(split_seed)
    train_i, val_i, test_i = _split_811(n, rng_split)

    payloads = [
        (Path(train_path), train_i),
        (Path(val_path), val_i),
        (Path(test_path), test_i),
    ]
    for path, idx in payloads:
        print(
            f'[build_bfm_nose_pt_cache] {path.name}: packing {len(idx)} samples ...',
            flush=True,
        )
        pack = _pack_split_items(
            pairs,
            idx,
            region_idx,
            num_workers=num_workers,
            target_root=tgt_r,
            progress_desc=path.stem,
        )
        pack['meta'] = {
            'split_indices': idx.astype(np.int64),
            'n_pairs_total': n,
            'regions': list(regions),
            'pairing_seed': pairing_seed,
            'split_seed': split_seed,
            'num_workers': num_workers,
            'target_root': str(tgt_r.resolve()),
        }
        torch.save(pack, path)
        print(f'saved {path} ({len(pack["items"])} samples)')


class BFMDataset(Dataset):
    """BFM 源/目标网格数据集。

    每个样本含 ``target_pose``：来自 ``target_root`` 下文件名含 ``pose`` 的 ``.npz``，
    由其中 ``trans``、``angle`` 展平后 ``concat`` 得到的一维向量。``packed_pt`` 模式会在
    ``meta.target_root`` 或构造参数 ``target_root`` 对应目录中解析 npz（与 OBJ 配对 stem）。
    """

    def __init__(
        self,
        packed_pt: Optional[Union[str, Path]] = None,
        source_root: Union[str, Path] = CREMAD_root,
        target_root: Union[str, Path] = MEAD_root,
        regions: Optional[Sequence[NoseRegion]] = None,
        transform: Optional[Callable] = None,
        extension: str = '.obj',
        seed: Optional[int] = None,
    ):
        self.transform = transform

        if packed_pt is not None:
            path = Path(packed_pt)
            if not path.is_file():
                raise FileNotFoundError(f'packed_pt not found: {path}')
            try:
                data = torch.load(path, map_location='cpu', weights_only=False)
            except TypeError:
                data = torch.load(path, map_location='cpu')
            if 'items' not in data or 'faces' not in data:
                raise ValueError(f'{path}: expected keys "items", "faces"')
            self._packed_items: list = data['items']
            self._packed_faces: torch.Tensor = data['faces']
            self._packed_mode = True
            self.regions = tuple(data.get('meta', {}).get('regions', []))
            self.source_root = None
            self.target_root = None
            self._region_face_indices = None
            self._samples = None
            tr = data.get('meta', {}).get('target_root')
            if tr is not None:
                self._packed_target_root = Path(tr)
            else:
                self._packed_target_root = Path(target_root)
            return

        self._packed_mode = False
        self._packed_items = None
        self._packed_faces = None
        self._packed_target_root = None
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        if regions is None:
            self.regions = ()
        else:
            self.regions = tuple(regions)

        if not self.source_root.is_dir():
            raise FileNotFoundError(f'source_root not found: {self.source_root}')
        if not self.target_root.is_dir():
            raise FileNotFoundError(f'target_root not found: {self.target_root}')

        ext = extension if extension.startswith('.') else f'.{extension}'
        self._samples = _pair_paths_from_roots(
            self.source_root, self.target_root, ext, seed
        )

        if not self.regions:
            self._region_face_indices = None
        else:
            self._region_face_indices = _combine_region_face_indices(self.regions)

        self._init_runtime_topology_cache()

    def _init_runtime_topology_cache(self) -> None:
        assert self._samples is not None
        first_src = self._samples[0][0]
        obj0 = read_obj(str(first_src), tri=True)
        faces_full = np.asarray(obj0.fvs, dtype=np.int64)
        if faces_full.ndim != 2 or faces_full.shape[1] != 3:
            raise ValueError(
                f'{first_src}: expected triangle faces (F, 3), got shape {faces_full.shape}'
            )
        self._faces_full = faces_full
        if self._region_face_indices is not None:
            vtx_pick, sub_f = _submesh_vertex_pick_and_faces(
                faces_full, self._region_face_indices
            )
            self._vtx_pick = vtx_pick
            self._faces_torch = torch.from_numpy(np.ascontiguousarray(sub_f))
        else:
            self._vtx_pick = None
            self._faces_torch = torch.from_numpy(np.ascontiguousarray(faces_full))

    def __len__(self) -> int:
        if self._packed_mode:
            return len(self._packed_items)
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        if self._packed_mode:
            it = self._packed_items[idx]
            tp = it.get('target_pose')
            if tp is None and self._packed_target_root is not None:
                tgt_stem = it.get('target_stem') or it['stem'].split('__')[-1]
                pz = _resolve_pose_npz_for_target_stem(
                    tgt_stem, self._packed_target_root
                )
                tp = torch.from_numpy(
                    np.ascontiguousarray(_load_target_pose_from_npz(pz))
                )
            if tp is None:
                raise ValueError(
                    f'样本 {idx} 缺少 target_pose：缓存项内无 target_pose 且无法从 '
                    f'{self._packed_target_root!s} 解析 pose npz；请检查 stem/target_stem 命名或重建 .pt'
                )
            sample = {
                'source_verts': it['source_verts'],
                'target_verts': it['target_verts'],
                'target_pose': tp,
                'faces': self._packed_faces,
                'stem': it['stem'],
            }
        else:
            assert self._samples is not None
            source_path, target_path = self._samples[idx]
            obj_s = read_obj(str(source_path), tri=True)
            obj_t = read_obj(str(target_path), tri=True)
            source_verts = np.asarray(obj_s.vs, dtype=np.float32)
            target_verts = np.asarray(obj_t.vs, dtype=np.float32)
            faces = np.asarray(obj_s.fvs, dtype=np.int64)
            faces_t = np.asarray(obj_t.fvs, dtype=np.int64)

            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(
                    f'{source_path}: expected triangle faces (F, 3), got shape {faces.shape}'
                )
            if source_verts.shape[0] != target_verts.shape[0]:
                raise ValueError(
                    f'{source_path.stem}: vertex count mismatch '
                    f'source {source_verts.shape[0]} vs target {target_verts.shape[0]}'
                )
            if faces.shape != self._faces_full.shape or not np.array_equal(
                faces, self._faces_full
            ):
                raise ValueError(
                    f'{source_path}: face indices differ from dataset reference topology'
                )
            if faces_t.shape != self._faces_full.shape or not np.array_equal(
                faces_t, self._faces_full
            ):
                raise ValueError(
                    f'{target_path}: face indices differ from dataset reference topology'
                )

            vmax = int(faces.max()) if faces.size else -1
            if vmax >= source_verts.shape[0]:
                raise ValueError(
                    f'{source_path}: face index max {vmax} >= num verts {source_verts.shape[0]}'
                )

            if self._vtx_pick is not None:
                source_verts = source_verts[self._vtx_pick]
                target_verts = target_verts[self._vtx_pick]

            stem = f'{source_path.stem}__{target_path.stem}'
            pz = _resolve_pose_npz_for_target_stem(target_path.stem, self.target_root)
            target_pose = torch.from_numpy(
                np.ascontiguousarray(_load_target_pose_from_npz(pz))
            )
            sample = {
                'source_verts': torch.from_numpy(np.ascontiguousarray(source_verts)),
                'target_verts': torch.from_numpy(np.ascontiguousarray(target_verts)),
                'target_pose': target_pose,
                'faces': self._faces_torch,
                'stem': stem,
            }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    import polyscope as ps

    if not TRAIN_BFM_NOSE_PT.is_file():
        build_bfm_nose_pt_cache(pairing_seed=0, split_seed=0, num_workers=8)

    dataset = BFMDataset(packed_pt=TRAIN_BFM_NOSE_PT)
    print('len', len(dataset))

    s0 = dataset[0]
    print('stem', s0['stem'])
    print('source_verts', s0['source_verts'].shape, s0['source_verts'].dtype)
    print('target_verts', s0['target_verts'].shape, s0['target_verts'].dtype)
    print('faces', s0['faces'].shape, s0['faces'].dtype)
    print('target_pose', s0['target_pose'].shape, s0['target_pose'].dtype)

    ps.init()
    ps.register_surface_mesh(
        'nose_source',
        vertices=s0['source_verts'].cpu().numpy(),
        faces=s0['faces'].cpu().numpy(),
        enabled=True,
        color=(0.75, 0.75, 0.75),
        edge_width=0.95,
        material='clay',
        smooth_shade=False,
        back_face_policy='custom',
    )
    bias = np.array([0.5, 0.0, 0.0])
    ps.register_surface_mesh(
        'nose_target',
        vertices=s0['target_verts'].cpu().numpy() + bias,
        faces=s0['faces'].cpu().numpy(),
        enabled=True,
        color=(0.75, 0.75, 0.75),
        edge_width=0.95,
        material='clay',
        smooth_shade=False,
        back_face_policy='custom',
    )
    ps.show()
