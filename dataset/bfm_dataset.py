import multiprocessing as mp
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mesh import read_obj


_DATASET_DIR = Path(__file__).resolve().parent
CREMAD_root = '/media/ubuntu/SSD/PHACK_data/bfm_meshes/cremad'
MEAD_root = '/media/ubuntu/SSD/PHACK_data/bfm_meshes/mead'
FFHQ_root = '/media/ubuntu/SSD/PHACK_data/bfm_meshes/ffhq'

TRAIN_BFM_PT = _DATASET_DIR / 'train_bfm.pt'
VAL_BFM_PT = _DATASET_DIR / 'val_bfm.pt'
TEST_BFM_PT = _DATASET_DIR / 'test_bfm.pt'

# 旧脚本可能仍引用 nose 命名的路径常量；与新 .pt 对齐
TRAIN_BFM_NOSE_PT = TRAIN_BFM_PT
VAL_BFM_NOSE_PT = VAL_BFM_PT
TEST_BFM_NOSE_PT = TEST_BFM_PT


def _resolve_pose_npz_for_obj(obj_path: Path) -> Path:
    """与 ``obj_path`` 同目录下解析 pose 的 ``.npz``（与 bfm_pair 命名规则兼容）。"""
    d = obj_path.parent
    stem = obj_path.stem
    primary = d / (stem.replace('_mid_mesh', '_pose') + '.npz')
    if primary.is_file():
        return primary
    alt = d / f'{stem}_pose.npz'
    if alt.is_file():
        return alt
    cands = sorted(
        p for p in d.glob('*.npz') if 'pose' in p.name.lower()
    )
    if not cands:
        raise FileNotFoundError(
            f'未在 {d} 下找到与 {obj_path.name} 对应的 pose .npz（已尝试 '
            f'{primary.name}、{alt.name} 及含 pose 的 npz）'
        )
    if len(cands) == 1:
        return cands[0]
    stem_low = stem.lower()
    matched = [p for p in cands if stem_low in p.stem.lower()]
    if len(matched) == 1:
        return matched[0]
    raise FileNotFoundError(
        f'{d}: 多个含 pose 的 npz，无法唯一对应 {obj_path.name}: '
        f'{[p.name for p in cands]}'
    )


def _load_pose_from_npz(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path)
    if 'trans' not in z.files:
        raise KeyError(f'{npz_path}: npz 缺少数组 trans')
    if 'angle' not in z.files:
        raise KeyError(f'{npz_path}: npz 缺少数组 angle')
    trans = np.asarray(z['trans'], dtype=np.float32).reshape(-1)
    angle = np.asarray(z['angle'], dtype=np.float32).reshape(-1)
    return np.concatenate([trans, angle], axis=0)


def _read_obj_verts_tris_fast(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """逐行流式读取 OBJ：只解析 ``v`` 与 ``f``，多边形面按与 ``read_obj(..., tri=True)`` 相同的扇形三角化。

    ``utils.mesh.read_obj`` 会一次性读入整个文件再按行拆分，大网格极慢；本函数用于建缓存批量读。
    """
    vs: List[List[float]] = []
    fvs: List[List[int]] = []
    with open(obj_path, 'r', encoding='utf-8', errors='replace', newline='') as fp:
        for line in fp:
            if line.startswith('v '):
                vs.append(list(map(float, line[2:].strip().split())))
            elif line.startswith('f '):
                fv: List[int] = []
                for tok in line[2:].strip().split():
                    vid = tok.split('/', 1)[0]
                    if vid == '':
                        continue
                    fv.append(int(vid) - 1)
                if len(fv) < 3:
                    continue
                for k in range(2, len(fv)):
                    fvs.append([fv[0], fv[k - 1], fv[k]])
    verts = np.asarray(vs, dtype=np.float32)
    faces = np.asarray(fvs, dtype=np.int64)
    return verts, faces


def _numpy_pack_to_torch(item: Dict) -> Dict:
    """进程池/numpy 路径下转为与缓存一致的 torch 字典。"""
    v = item['verts']
    if isinstance(v, torch.Tensor):
        return item
    return {
        'verts': torch.from_numpy(np.ascontiguousarray(v)),
        'pose': torch.from_numpy(np.ascontiguousarray(item['pose'])),
        'stem': item['stem'],
        'obj_path': item['obj_path'],
    }


def _collect_obj_paths(
    cremad_root: Path,
    mead_root: Path,
    ffhq_root: Path,
    ext: str,
) -> List[Path]:
    paths: List[Path] = []
    for root in (cremad_root, mead_root, ffhq_root):
        if not root.is_dir():
            raise FileNotFoundError(f'网格根目录不存在: {root}')
        paths.extend(
            sorted(root.glob(f'*{ext}'), key=lambda p: p.as_posix())
        )
    return paths


def _load_mesh_item(
    obj_path: Path,
    faces_ref: np.ndarray,
    *,
    use_fast_obj_parser: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    if use_fast_obj_parser:
        verts, faces = _read_obj_verts_tris_fast(str(obj_path))
    else:
        obj = read_obj(str(obj_path), tri=True)
        verts = np.asarray(obj.vs, dtype=np.float32)
        faces = np.asarray(obj.fvs, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f'{obj_path}: 需要三角面 (F, 3)，当前 shape {faces.shape}'
        )
    if faces.shape != faces_ref.shape or not np.array_equal(faces, faces_ref):
        raise ValueError(
            f'{obj_path.stem}: 面索引与参考拓扑不一致'
        )
    vmax = int(faces.max()) if faces.size else -1
    if vmax >= verts.shape[0]:
        raise ValueError(
            f'{obj_path}: 面最大顶点索引 {vmax} >= 顶点数 {verts.shape[0]}'
        )
    pose_npz = _resolve_pose_npz_for_obj(obj_path)
    pose = _load_pose_from_npz(pose_npz)
    return verts, faces, obj_path.stem, pose


_POOL_WORKER_FACES: Optional[np.ndarray] = None


def _pool_worker_init(faces_ref: np.ndarray) -> None:
    """每个 Pool worker 进程调用一次；共享参考面拓扑（只读）。"""
    global _POOL_WORKER_FACES
    _POOL_WORKER_FACES = np.ascontiguousarray(faces_ref)


def _pool_worker_one(task: Tuple[int, str]) -> Tuple[int, Optional[Dict], Optional[str]]:
    """Pool worker：处理单个 ``(idx, obj_path_str)``。"""
    global _POOL_WORKER_FACES
    assert _POOL_WORKER_FACES is not None
    idx, path_str = task
    try:
        p = Path(path_str)
        verts, _, stem, pose = _load_mesh_item(p, _POOL_WORKER_FACES)
        out: Dict = {
            'verts': verts,
            'pose': pose,
            'stem': stem,
            'obj_path': str(p.resolve()),
        }
        return idx, out, None
    except BaseException as e:
        return idx, None, f'{type(e).__name__}: {e}'


def _pack_all_obj_items_pool(
    obj_paths: Sequence[Path],
    faces_ref: np.ndarray,
    num_workers: int,
    progress_desc: str,
) -> Tuple[List[Optional[dict]], List[Dict[str, str]]]:
    """进程池并行装载（``read_timeout_sec==0`` 时使用）。Linux 上一般为 fork，启动成本低。"""
    n = len(obj_paths)
    rows: List[Optional[dict]] = [None] * n
    skipped: List[Dict[str, str]] = []
    nw = max(1, int(num_workers))
    tasks = [(i, str(obj_paths[i])) for i in range(n)]
    # 减轻 IPC 往返：块越大单次传越多任务名，总轮次越少
    chunksize = max(8, min(512, n // max(nw * 2, 1) or 8))
    pool_kw: Dict[str, Union[int, None]] = {}
    if n >= 8192:
        mt = max(256, min(1024, (max(1, n // max(nw, 1)) // 4) or 256))
        pool_kw['maxtasksperchild'] = mt

    ctx = mp.get_context()
    with ctx.Pool(
        processes=nw,
        initializer=_pool_worker_init,
        initargs=(faces_ref,),
        **pool_kw,
    ) as pool:
        it = pool.imap_unordered(_pool_worker_one, tasks, chunksize=chunksize)
        for idx_r, item, err in tqdm(
            it,
            total=n,
            desc=progress_desc,
            mininterval=0.0,
            smoothing=0,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            if item is not None:
                rows[idx_r] = _numpy_pack_to_torch(item)
            else:
                skipped.append({
                    'path': str(obj_paths[idx_r]),
                    'reason': err or 'unknown',
                })

    return rows, skipped


def _force_terminate_proc(proc: mp.Process, join_sec: float = 20.0) -> None:
    if not proc.is_alive():
        proc.join(timeout=0.5)
        return
    proc.terminate()
    proc.join(join_sec)
    if proc.is_alive():
        proc.kill()
        proc.join(10.0)


def _safe_close_conn(conn: Optional[Connection]) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


def _spawn_child_first_faces(child_w: Connection, path_str: str) -> None:
    """子进程：仅取拓扑三角面，用于解析参考面。"""
    try:
        _, faces = _read_obj_verts_tris_fast(str(path_str))
        child_w.send(('ok', faces))
    except BaseException as e:
        try:
            child_w.send(('err', f'{type(e).__name__}: {e}'))
        except Exception:
            pass
    finally:
        _safe_close_conn(child_w)


def _spawn_child_pack_one(
    child_w: Connection,
    faces_ref: np.ndarray,
    obj_path_str: str,
    idx: int,
) -> None:
    """独立 ``spawn`` 子进程内整段装载；经 ``Pipe`` 回传。"""
    try:
        p = Path(obj_path_str)
        verts, _, stem, pose = _load_mesh_item(p, faces_ref)
        out: Dict = {
            'verts': verts,
            'pose': pose,
            'stem': stem,
            'obj_path': str(p.resolve()),
        }
        child_w.send((idx, out, None))
    except BaseException as e:
        try:
            child_w.send((idx, None, f'{type(e).__name__}: {e}'))
        except Exception:
            pass
    finally:
        _safe_close_conn(child_w)


def _mp_spawn_ctx() -> mp.context.BaseContext:
    return mp.get_context('spawn')


def _first_valid_faces_ref(
    paths: Sequence[Path],
    read_timeout_sec: float,
) -> Tuple[np.ndarray, Path]:
    """从 ``paths`` 依次尝试得到首个三角面参考拓扑；``read_timeout_sec>0`` 时每步在独立子进程中读并强杀超时。"""
    last_err: Optional[BaseException] = None
    if read_timeout_sec <= 0:
        for p in paths:
            try:
                _, faces = _read_obj_verts_tris_fast(str(p))
                if faces.ndim == 2 and faces.shape[1] == 3:
                    return faces, p
            except BaseException as e:
                last_err = e
                continue
    else:
        ctx = _mp_spawn_ctx()
        poll_wait = float(read_timeout_sec) + 15.0
        for p in paths:
            parent_r, child_w = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_spawn_child_first_faces,
                args=(child_w, str(p)),
            )
            proc.start()
            _safe_close_conn(child_w)
            try:
                if parent_r.poll(poll_wait):
                    tag, payload = parent_r.recv()
                else:
                    _force_terminate_proc(proc)
                    last_err = TimeoutError(
                        f'读取参考拓扑壁钟超时 ({read_timeout_sec}s): {p}'
                    )
                    _safe_close_conn(parent_r)
                    continue
            except (EOFError, BrokenPipeError, OSError, ConnectionRefusedError) as e:
                _force_terminate_proc(proc)
                last_err = e
                _safe_close_conn(parent_r)
                continue
            _force_terminate_proc(proc)
            _safe_close_conn(parent_r)
            if tag == 'ok':
                faces = payload
                if isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3:
                    return faces, Path(str(p))
                last_err = ValueError(
                    f'参考 OBJ 非三角面网格: {p} shape={getattr(faces, "shape", None)}'
                )
            else:
                last_err = RuntimeError(str(payload))

    if last_err is not None:
        raise ValueError('无法从任何 OBJ 得到参考三角面网格') from last_err
    raise ValueError('无法从任何 OBJ 得到参考三角面网格')


def _pack_all_obj_items(
    obj_paths: Sequence[Path],
    faces_ref: np.ndarray,
    num_workers: int,
    read_timeout_sec: float,
    progress_desc: str = 'pack_all',
) -> Tuple[List[Optional[dict]], List[Dict[str, str]]]:
    """装载全部 OBJ。``read_timeout_sec>0`` 时每个文件在独立子进程中读，超时则 ``terminate/kill`` 并跳过。"""
    n = len(obj_paths)
    rows: List[Optional[dict]] = [None] * n
    skipped: List[Dict[str, str]] = []

    if read_timeout_sec <= 0:
        if max(1, int(num_workers)) > 1 and n > 1:
            return _pack_all_obj_items_pool(
                obj_paths, faces_ref, num_workers, progress_desc,
            )
        for i in tqdm(
            range(n),
            desc=progress_desc,
            mininterval=0.0,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            try:
                verts, _, stem, pose = _load_mesh_item(Path(str(obj_paths[i])), faces_ref)
                rows[i] = {
                    'verts': torch.from_numpy(np.ascontiguousarray(verts)),
                    'pose': torch.from_numpy(np.ascontiguousarray(pose)),
                    'stem': stem,
                    'obj_path': str(Path(obj_paths[i]).resolve()),
                }
            except BaseException as e:
                skipped.append({
                    'path': str(obj_paths[i]),
                    'reason': f'{type(e).__name__}: {e}',
                })
        return rows, skipped

    ctx = _mp_spawn_ctx()
    wall_limit = float(read_timeout_sec) + 20.0
    max_parallel = max(1, int(num_workers))

    active: List[Tuple[mp.Process, Connection, int, float]] = []
    next_idx = 0
    completed = 0

    pbar = tqdm(
        total=n,
        desc=progress_desc,
        mininterval=0.0,
        smoothing=0,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    try:
        while completed < n:
            while len(active) < max_parallel and next_idx < n:
                parent_r, child_w = ctx.Pipe(duplex=False)
                proc = ctx.Process(
                    target=_spawn_child_pack_one,
                    args=(child_w, faces_ref, str(obj_paths[next_idx]), next_idx),
                )
                proc.start()
                _safe_close_conn(child_w)
                active.append((proc, parent_r, next_idx, time.monotonic()))
                next_idx += 1

            if not active:
                break

            progressed = False
            now = time.monotonic()

            for k in range(len(active) - 1, -1, -1):
                proc, parent_r, idx, t0 = active[k]
                if now - t0 > wall_limit:
                    _force_terminate_proc(proc)
                    _safe_close_conn(parent_r)
                    rows[idx] = None
                    skipped.append({
                        'path': str(obj_paths[idx]),
                        'reason': f'wall_timeout>{read_timeout_sec}s',
                    })
                    active.pop(k)
                    completed += 1
                    pbar.update(1)
                    progressed = True
                    continue

                if not parent_r.poll(0):
                    continue

                try:
                    msg = parent_r.recv()
                except (EOFError, BrokenPipeError, OSError, ConnectionRefusedError) as e:
                    _force_terminate_proc(proc)
                    _safe_close_conn(parent_r)
                    skipped.append({
                        'path': str(obj_paths[idx]),
                        'reason': f'pipe_recv_error:{type(e).__name__}: {e}',
                    })
                    active.pop(k)
                    completed += 1
                    pbar.update(1)
                    progressed = True
                    continue

                _safe_close_conn(parent_r)
                idx_r, item, err = msg
                _force_terminate_proc(proc)
                if item is not None:
                    rows[idx_r] = _numpy_pack_to_torch(item)
                else:
                    skipped.append({
                        'path': str(obj_paths[idx_r]),
                        'reason': err or 'unknown',
                    })
                active.pop(k)
                completed += 1
                pbar.update(1)
                progressed = True

            if not progressed:
                time.sleep(0.02)
    finally:
        for proc, parent_r, _, _ in list(active):
            _force_terminate_proc(proc)
            _safe_close_conn(parent_r)
        pbar.close()

    return rows, skipped


def _split_811(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError('划分需要至少 1 个样本')
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


def build_bfm_pt_cache(
    cremad_root: Union[str, Path] = CREMAD_root,
    mead_root: Union[str, Path] = MEAD_root,
    ffhq_root: Union[str, Path] = FFHQ_root,
    split_seed: int = 0,
    out_dir: Union[str, Path] = _DATASET_DIR,
    extension: str = '.obj',
    train_path: Union[str, Path] = TRAIN_BFM_PT,
    val_path: Union[str, Path] = VAL_BFM_PT,
    test_path: Union[str, Path] = TEST_BFM_PT,
    num_workers: int = 4,
    read_timeout_sec: float = 0.0,
) -> None:
    """
    将 CREMAD、MEAD、FFHQ 目录下 ``*.obj`` 与对应 pose ``.npz`` 打包并按 8:1:1 写入 ``.pt``。

    - **默认** ``read_timeout_sec=0``：使用 **多进程 Pool**（Linux 上多为 fork）并行读盘，
      ``faces_ref`` 每个 worker 只初始化一次，适合数万级 OBJ，比「每文件 spawn」快一个数量级以上。
    - ``read_timeout_sec>0``：改为 **每文件独立 spawn + Pipe**，父进程壁钟超时后可 ``terminate/kill``
      该子进程（极慢，但可避开卡死在 ``read_obj`` 里的坏文件）。

    若在主进程已初始化 CUDA / 复杂线程环境，Linux fork 可能不稳定，可仅开单进程 ``num_workers=1``
    或改用 ``read_timeout_sec>0`` 的隔离模式。
    """
    cr = Path(cremad_root)
    md = Path(mead_root)
    ffhq = Path(ffhq_root)
    out_d = Path(out_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    ext = extension if extension.startswith('.') else f'.{extension}'
    all_paths = _collect_obj_paths(cr, md, ffhq, ext)
    if not all_paths:
        raise ValueError(f'未在 {cr}, {md} 与 {ffhq} 下找到 *{ext}')

    cr_r, md_r, ff_r = cr.resolve(), md.resolve(), ffhq.resolve()
    n_cremad_raw = sum(1 for p in all_paths if p.parent.resolve() == cr_r)
    n_mead_raw = sum(1 for p in all_paths if p.parent.resolve() == md_r)
    n_ffhq_raw = sum(1 for p in all_paths if p.parent.resolve() == ff_r)
    print(
        f'[build_bfm_pt_cache] 磁盘上 {ext} 个数 — CREMAD: {n_cremad_raw}, MEAD: {n_mead_raw}, '
        f'FFHQ: {n_ffhq_raw}，合计: {len(all_paths)}',
        flush=True,
    )

    nw = max(1, int(num_workers))
    load_mode = (
        'process_pool'
        if read_timeout_sec <= 0 and nw > 1 and len(all_paths) > 1
        else ('spawn_subprocess' if read_timeout_sec > 0 else 'inprocess')
    )
    print(
        f'[build_bfm_pt_cache] 加载模式: {load_mode}（workers={nw}, read_timeout_sec={read_timeout_sec}）',
        flush=True,
    )

    faces_ref, ref_path = _first_valid_faces_ref(all_paths, read_timeout_sec)
    rows, skipped_records = _pack_all_obj_items(
        all_paths,
        faces_ref,
        num_workers=num_workers,
        read_timeout_sec=read_timeout_sec,
        progress_desc='load_all',
    )

    valid_paths: List[Path] = []
    valid_items: List[dict] = []
    for i, p in enumerate(all_paths):
        it = rows[i]
        if it is not None:
            valid_paths.append(p)
            valid_items.append(it)

    n_skip = len(skipped_records)
    if n_skip:
        print(f'[build_bfm_pt_cache] 跳过 {n_skip} 个样本（超时或读取异常），不写入 .pt', flush=True)
        for rec in skipped_records[:20]:
            print(f"  skip: {rec['path']} ({rec['reason']})", flush=True)
        if n_skip > 20:
            print(f'  ... 另有 {n_skip - 20} 条省略', flush=True)

    if not valid_paths:
        raise ValueError('所有 OBJ 均装载失败（超时或异常），无法生成缓存')

    n_cremad = sum(1 for p in valid_paths if p.parent.resolve() == cr_r)
    n_mead = sum(1 for p in valid_paths if p.parent.resolve() == md_r)
    n_ffhq = sum(1 for p in valid_paths if p.parent.resolve() == ff_r)
    print(
        f'[build_bfm_pt_cache] 成功装载 {ext} — CREMAD: {n_cremad}, MEAD: {n_mead}, '
        f'FFHQ: {n_ffhq}，合计: {len(valid_paths)}',
        flush=True,
    )

    n = len(valid_paths)
    rng_split = np.random.default_rng(split_seed)
    train_i, val_i, test_i = _split_811(n, rng_split)

    payloads = [
        (Path(train_path), train_i),
        (Path(val_path), val_i),
        (Path(test_path), test_i),
    ]
    for path, idx in tqdm(payloads):
        print(
            f'[build_bfm_pt_cache] {path.name}: 写入 {len(idx)} 个样本 ...',
            flush=True,
        )
        items_split = [valid_items[int(j)] for j in idx]
        pack = {
            'faces': torch.from_numpy(np.ascontiguousarray(faces_ref)),
            'items': items_split,
        }
        pack['meta'] = {
            'split_indices': idx.astype(np.int64),
            'n_samples_total': n,
            'n_candidates': len(all_paths),
            'n_skipped': len(skipped_records),
            'skipped_samples': skipped_records,
            'split_seed': split_seed,
            'cremad_root': str(cr.resolve()),
            'mead_root': str(md.resolve()),
            'ffhq_root': str(ffhq.resolve()),
            'reference_obj': str(ref_path.resolve()),
            'num_workers': num_workers,
            'read_timeout_sec': read_timeout_sec,
            'load_isolation': load_mode,
            'fast_obj_reader': True,
        }
        torch.save(pack, path)
        print(f'saved {path} ({len(pack["items"])} samples)')


def build_bfm_nose_pt_cache(**kwargs) -> None:
    """兼容旧函数名：与 :func:`build_bfm_pt_cache` 相同。"""
    build_bfm_pt_cache(**kwargs)


class BFMDataset(Dataset):
    """BFM 整头网格：每样本为单个 OBJ 的顶点与对应 ``pose``（``trans``+``angle`` 展平）。"""

    def __init__(
        self,
        packed_pt: Optional[Union[str, Path]] = None,
        cremad_root: Union[str, Path] = CREMAD_root,
        mead_root: Union[str, Path] = MEAD_root,
        ffhq_root: Union[str, Path] = FFHQ_root,
        transform: Optional[Callable] = None,
        extension: str = '.obj',
    ):
        self.transform = transform

        if packed_pt is not None:
            path = Path(packed_pt)
            if not path.is_file():
                raise FileNotFoundError(f'packed_pt 不存在: {path}')
            try:
                data = torch.load(path, map_location='cpu', weights_only=False)
            except TypeError:
                data = torch.load(path, map_location='cpu')
            if 'items' not in data or 'faces' not in data:
                raise ValueError(f'{path}: 需要包含键 items、faces')
            self._packed_items: list = data['items']
            self._packed_faces: torch.Tensor = data['faces']
            self._packed_mode = True
            self._obj_paths = None
            self._faces_ref_np = None
            return

        self._packed_mode = False
        self._packed_items = None
        self._packed_faces = None
        cr = Path(cremad_root)
        md = Path(mead_root)
        ffhq = Path(ffhq_root)
        if not cr.is_dir():
            raise FileNotFoundError(f'cremad_root 不存在: {cr}')
        if not md.is_dir():
            raise FileNotFoundError(f'mead_root 不存在: {md}')
        if not ffhq.is_dir():
            raise FileNotFoundError(f'ffhq_root 不存在: {ffhq}')
        ext = extension if extension.startswith('.') else f'.{extension}'
        self._obj_paths = _collect_obj_paths(cr, md, ffhq, ext)
        if not self._obj_paths:
            raise ValueError(f'未在 {cr}、{md} 与 {ffhq} 下找到 *{ext}')
        _, self._faces_ref_np = _read_obj_verts_tris_fast(str(self._obj_paths[0]))

    def __len__(self) -> int:
        if self._packed_mode:
            return len(self._packed_items)
        return len(self._obj_paths)  # type: ignore[arg-type]

    @staticmethod
    def _item_dict(
        verts: torch.Tensor,
        pose: torch.Tensor,
        faces: torch.Tensor,
        stem: str,
    ) -> dict:
        sample = {
            'verts': verts,
            'pose': pose,
            'faces': faces,
            'stem': stem,
            'source_verts': verts,
            'target_pose': pose,
        }
        return sample

    def __getitem__(self, idx: int) -> dict:
        if self._packed_mode:
            it = self._packed_items[idx]
            sample = self._item_dict(
                it['verts'],
                it['pose'],
                self._packed_faces,
                it['stem'],
            )
        else:
            assert self._obj_paths is not None and self._faces_ref_np is not None
            p = self._obj_paths[idx]
            v, _, stem, pose = _load_mesh_item(p, self._faces_ref_np)
            sample = self._item_dict(
                torch.from_numpy(np.ascontiguousarray(v)),
                torch.from_numpy(np.ascontiguousarray(pose)),
                torch.from_numpy(np.ascontiguousarray(self._faces_ref_np)),
                stem,
            )

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':

    import polyscope as ps

    # if not TRAIN_BFM_PT.is_file():
        # build_bfm_pt_cache(split_seed=0, num_workers=8)

    # build_bfm_pt_cache(split_seed=0, num_workers=8)

    dataset = BFMDataset(packed_pt=TRAIN_BFM_PT)
    print('len', len(dataset))

    s0 = dataset[0]
    print('stem', s0['stem'])
    print('verts', s0['verts'].shape, s0['verts'].dtype)
    print('faces', s0['faces'].shape, s0['faces'].dtype)
    print('pose', s0['pose'].shape, s0['pose'].dtype)

    ps.init()
    ps.register_surface_mesh(
        'bfm_mesh',
        vertices=s0['verts'].cpu().numpy(),
        faces=s0['faces'].cpu().numpy(),
        enabled=True,
        color=(0.75, 0.75, 0.75),
        edge_width=0.95,
        material='clay',
        smooth_shade=False,
        back_face_policy='custom',
    )
    ps.show()
