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
FACESCAPE_root = '/media/ubuntu/xb/FaceScape_Dataset/HACK_fit_repaired'
IMHEAD_root = '/media/ubuntu/xb/ImHead_dataset/HACK_fit_repaired'
FFHQ_root = '/media/ubuntu/xb/FFHQ_Dataset/phack_refine_repair'
FFHQ_pose_root = '/media/ubuntu/xb/FFHQ_Dataset/phack_raw'

TRAIN_HACK_PT = _DATASET_DIR / 'train_hack.pt'
VAL_HACK_PT = _DATASET_DIR / 'val_hack.pt'
TEST_HACK_PT = _DATASET_DIR / 'test_hack.pt'
TRAIN_BFM_PT = TRAIN_HACK_PT
VAL_BFM_PT = VAL_HACK_PT
TEST_BFM_PT = TEST_HACK_PT

SOURCE_FACESCAPE = 'FaceScape'
SOURCE_IMHEAD = 'ImHead'
SOURCE_FFHQ = 'FFHQ'
HACK_SOURCE_LABELS = (SOURCE_FACESCAPE, SOURCE_IMHEAD, SOURCE_FFHQ)
HACK_SOURCE_TO_ID = {name: i for i, name in enumerate(HACK_SOURCE_LABELS)}


def _path_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _source_label_for_path(
    obj_path: Union[str, Path],
    facescape_root: Path,
    imhead_root: Path,
    ffhq_root: Path,
) -> str:
    """根据 OBJ 路径判定所属数据源（FaceScape / ImHead / FFHQ）。"""
    p = Path(obj_path).resolve()
    fs_r, md_r, ff_r = (
        facescape_root.resolve(),
        imhead_root.resolve(),
        ffhq_root.resolve(),
    )
    if _path_under_root(p, fs_r):
        return SOURCE_FACESCAPE
    if _path_under_root(p, md_r):
        return SOURCE_IMHEAD
    if _path_under_root(p, ff_r):
        return SOURCE_FFHQ
    raise ValueError(f'无法判定数据源: {obj_path}')


def _infer_source_from_obj_path(obj_path: Union[str, Path]) -> Optional[str]:
    """旧缓存无 ``source`` 字段时，从路径字符串启发式推断。"""
    s = str(obj_path).lower()
    if 'facescape' in s:
        return SOURCE_FACESCAPE
    if 'imhead' in s:
        return SOURCE_IMHEAD
    if 'ffhq' in s:
        return SOURCE_FFHQ
    return None


def _resolve_ffhq_phack_raw_npz(
    obj_path: Union[str, Path],
    ffhq_pose_root: Union[str, Path],
) -> Path:
    """FFHQ 修复网格 ``{id}_repair.obj`` → ``phack_raw/{id}_phack_raw.npz``。"""
    stem = Path(obj_path).stem
    if stem.endswith('_repair'):
        stem = stem[: -len('_repair')]
    elif stem.endswith('_phack_raw'):
        stem = stem[: -len('_phack_raw')]
    npz_path = Path(ffhq_pose_root) / f'{stem}_phack_raw.npz'
    if not npz_path.is_file():
        raise FileNotFoundError(
            f'未找到 FFHQ neck_pose npz: {npz_path}（对应 {obj_path}）'
        )
    return npz_path


def _load_neck_pose_from_npz(npz_path: Union[str, Path]) -> np.ndarray:
    """从 ``_phack_raw.npz`` 读取 neck pose，压缩为 6 维 float32 向量。

    磁盘上 ``neck_poses`` 一般为 (8, 3)，仅前两行非零；也兼容 ``neck_pose``、(2, 3) 与 (6,)。
    """
    z = np.load(npz_path)
    if 'neck_pose' in z.files:
        neck = np.asarray(z['neck_pose'], dtype=np.float32)
    elif 'neck_poses' in z.files:
        neck = np.asarray(z['neck_poses'], dtype=np.float32)
    else:
        raise KeyError(
            f'{npz_path}: npz 缺少 neck_pose / neck_poses，现有键 {list(z.files)}'
        )

    if neck.ndim == 2 and neck.shape == (8, 3):
        if not np.allclose(neck[2:], 0.0, atol=1e-6):
            raise ValueError(
                f'{npz_path}: neck_pose 期望 8x3 仅前两行非零，'
                f'但后 6 行 max|.|={float(np.abs(neck[2:]).max()):.6g}'
            )
        neck = neck[:2]
    elif neck.ndim == 2 and neck.shape == (2, 3):
        pass
    elif neck.ndim == 1 and neck.size == 6:
        return np.ascontiguousarray(neck, dtype=np.float32)
    else:
        raise ValueError(
            f'{npz_path}: neck_pose 形状应为 (8,3)/(2,3)/(6,)，当前 {neck.shape}'
        )
    return np.ascontiguousarray(neck.reshape(-1), dtype=np.float32)


def _load_ffhq_neck_pose_for_obj(
    obj_path: Union[str, Path],
    ffhq_pose_root: Union[str, Path],
) -> np.ndarray:
    npz_path = _resolve_ffhq_phack_raw_npz(obj_path, ffhq_pose_root)
    return _load_neck_pose_from_npz(npz_path)


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
    out = {
        'verts': torch.from_numpy(np.ascontiguousarray(v)),
        'stem': item['stem'],
        'obj_path': item['obj_path'],
    }
    if 'source' in item:
        out['source'] = item['source']
    if 'neck_pose' in item:
        neck = item['neck_pose']
        if not isinstance(neck, torch.Tensor):
            neck = torch.from_numpy(np.ascontiguousarray(neck))
        out['neck_pose'] = neck
    return out


def _collect_obj_paths(
    facescape_root: Path,
    imhead_root: Path,
    ffhq_root: Path,
) -> List[Path]:
    paths: List[Path] = []
    for root in (facescape_root, imhead_root, ffhq_root):
        if not root.is_dir():
            raise FileNotFoundError(f'网格根目录不存在: {root}')
        paths.extend(sorted(root.rglob('*.obj'), key=lambda p: p.as_posix()))
    return paths


def _faces_same_connectivity(faces: np.ndarray, faces_ref: np.ndarray) -> bool:
    """三角面连接关系是否一致（允许 OBJ 内 ``f`` 行顺序不同）。"""
    if faces.shape != faces_ref.shape:
        return False
    if np.array_equal(faces, faces_ref):
        return True
    a = np.sort(faces, axis=1)
    b = np.sort(faces_ref, axis=1)
    return bool(np.array_equal(np.sort(a, axis=0), np.sort(b, axis=0)))


def _load_mesh_item(
    obj_path: Path,
    faces_ref: np.ndarray,
    *,
    use_fast_obj_parser: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
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
    n_verts_ref = int(faces_ref.max()) + 1 if faces_ref.size else 0
    if verts.shape[0] != n_verts_ref:
        raise ValueError(
            f'{obj_path.stem}: 顶点数 {verts.shape[0]} != 参考 {n_verts_ref}'
        )
    if not _faces_same_connectivity(faces, faces_ref):
        raise ValueError(
            f'{obj_path.stem}: 面连接与参考拓扑不一致'
        )
    vmax = int(faces.max()) if faces.size else -1
    if vmax >= verts.shape[0]:
        raise ValueError(
            f'{obj_path}: 面最大顶点索引 {vmax} >= 顶点数 {verts.shape[0]}'
        )
    return verts, faces_ref, obj_path.stem


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
        verts, _, stem = _load_mesh_item(p, _POOL_WORKER_FACES)
        out: Dict = {
            'verts': verts,
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
        verts, _, stem = _load_mesh_item(p, faces_ref)
        out: Dict = {
            'verts': verts,
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
                verts, _, stem = _load_mesh_item(Path(str(obj_paths[i])), faces_ref)
                rows[i] = {
                    'verts': torch.from_numpy(np.ascontiguousarray(verts)),
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


def build_hack_pt_cache(
    facescape_root: Union[str, Path] = FACESCAPE_root,
    imhead_root: Union[str, Path] = IMHEAD_root,
    ffhq_root: Union[str, Path] = FFHQ_root,
    ffhq_pose_root: Union[str, Path] = FFHQ_pose_root,
    split_seed: int = 0,
    out_dir: Union[str, Path] = _DATASET_DIR,
    train_path: Union[str, Path] = TRAIN_HACK_PT,
    val_path: Union[str, Path] = VAL_HACK_PT,
    test_path: Union[str, Path] = TEST_HACK_PT,
    num_workers: int = 4,
    read_timeout_sec: float = 0.0,
) -> None:
    """
    将 FACESCAPE、IMHEAD、FFHQ 根目录下所有 ``.obj`` 网格打包并按 8:1:1 写入 ``.pt``。

    - **默认** ``read_timeout_sec=0``：使用 **多进程 Pool**（Linux 上多为 fork）并行读盘，
      ``faces_ref`` 每个 worker 只初始化一次，适合数万级 OBJ，比「每文件 spawn」快一个数量级以上。
    - ``read_timeout_sec>0``：改为 **每文件独立 spawn + Pipe**，父进程壁钟超时后可 ``terminate/kill``
      该子进程（极慢，但可避开卡死在 ``read_obj`` 里的坏文件）。

    若在主进程已初始化 CUDA / 复杂线程环境，Linux fork 可能不稳定，可仅开单进程 ``num_workers=1``
    或改用 ``read_timeout_sec>0`` 的隔离模式。
    """
    fs = Path(facescape_root)
    md = Path(imhead_root)
    ffhq = Path(ffhq_root)
    ffhq_pose = Path(ffhq_pose_root)
    out_d = Path(out_dir)
    out_d.mkdir(parents=True, exist_ok=True)
    if not ffhq_pose.is_dir():
        raise FileNotFoundError(f'FFHQ pose 根目录不存在: {ffhq_pose}')

    all_paths = _collect_obj_paths(fs, md, ffhq)
    if not all_paths:
        raise ValueError(f'未在 {fs}, {md}, {ffhq} 下找到任何 .obj 文件')

    fs_r, md_r, ff_r = fs.resolve(), md.resolve(), ffhq.resolve()
    n_facescape_raw = sum(1 for p in all_paths if _path_under_root(p, fs_r))
    n_imhead_raw = sum(1 for p in all_paths if _path_under_root(p, md_r))
    n_ffhq_raw = sum(1 for p in all_paths if _path_under_root(p, ff_r))
    print(
        f'[build_hack_pt_cache] 磁盘上 .obj 个数 — FACESCAPE: {n_facescape_raw}, IMHEAD: {n_imhead_raw}, '
        f'FFHQ: {n_ffhq_raw}, 合计: {len(all_paths)}',
        flush=True,
    )

    nw = max(1, int(num_workers))
    load_mode = (
        'process_pool'
        if read_timeout_sec <= 0 and nw > 1 and len(all_paths) > 1
        else ('spawn_subprocess' if read_timeout_sec > 0 else 'inprocess')
    )
    print(
        f'[build_hack_pt_cache] 加载模式: {load_mode}（workers={nw}, read_timeout_sec={read_timeout_sec}）',
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
    neck_pose_missing: List[str] = []
    for i, p in enumerate(all_paths):
        it = rows[i]
        if it is not None:
            valid_paths.append(p)
            it['source'] = _source_label_for_path(p, fs_r, md_r, ff_r)
            it['source_id'] = HACK_SOURCE_TO_ID[it['source']]
            if it['source'] == SOURCE_FFHQ:
                try:
                    it['neck_pose'] = _load_ffhq_neck_pose_for_obj(p, ffhq_pose)
                except (OSError, ValueError, KeyError) as e:
                    neck_pose_missing.append(f'{p} ({type(e).__name__}: {e})')
            valid_items.append(it)

    if neck_pose_missing:
        n_neck_miss = len(neck_pose_missing)
        print(
            f'[build_hack_pt_cache] 警告: {n_neck_miss} 个 FFHQ 样本缺少 neck_pose',
            flush=True,
        )
        for rec in neck_pose_missing[:10]:
            print(f'  neck_pose miss: {rec}', flush=True)
        if n_neck_miss > 10:
            print(f'  ... 另有 {n_neck_miss - 10} 条省略', flush=True)

    n_skip = len(skipped_records)
    if n_skip:
        print(f'[build_hack_pt_cache] 跳过 {n_skip} 个样本（超时或读取异常），不写入 .pt', flush=True)
        for rec in skipped_records[:20]:
            print(f"  skip: {rec['path']} ({rec['reason']})", flush=True)
        if n_skip > 20:
            print(f'  ... 另有 {n_skip - 20} 条省略', flush=True)

    if not valid_paths:
        raise ValueError('所有 OBJ 均装载失败（超时或异常），无法生成缓存')

    n_facescape = sum(1 for p in valid_paths if _path_under_root(p, fs_r))
    n_imhead = sum(1 for p in valid_paths if _path_under_root(p, md_r))
    n_ffhq = sum(1 for p in valid_paths if _path_under_root(p, ff_r))
    print(
        f'[build_hack_pt_cache] 成功装载 .obj — FACESCAPE: {n_facescape}, IMHEAD: {n_imhead}, '
        f'FFHQ: {n_ffhq}, 合计: {len(valid_paths)}',
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
            'facescape_root': str(fs.resolve()),
            'imhead_root': str(md.resolve()),
            'ffhq_root': str(ffhq.resolve()),
            'ffhq_pose_root': str(ffhq_pose.resolve()),
            'obj_glob': '*.obj',
            'reference_obj': str(ref_path.resolve()),
            'num_workers': num_workers,
            'read_timeout_sec': read_timeout_sec,
            'load_isolation': load_mode,
            'fast_obj_reader': True,
            'source_labels': list(HACK_SOURCE_LABELS),
            'source_to_id': dict(HACK_SOURCE_TO_ID),
        }
        torch.save(pack, path)
        print(f'saved {path} ({len(pack["items"])} samples)')


def build_bfm_nose_pt_cache(**kwargs) -> None:
    """兼容旧函数名：与 :func:`build_hack_pt_cache` 相同。"""
    build_hack_pt_cache(**kwargs)


class HACKDataset(Dataset):
    """HACK 整头网格：每样本为 ``.obj`` 顶点（及共享三角面拓扑）。"""

    def __init__(
        self,
        packed_pt: Optional[Union[str, Path]] = None,
        facescape_root: Union[str, Path] = FACESCAPE_root,
        imhead_root: Union[str, Path] = IMHEAD_root,
        ffhq_root: Union[str, Path] = FFHQ_root,
        ffhq_pose_root: Union[str, Path] = FFHQ_pose_root,
        sources: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        if sources is not None:
            unknown = set(sources) - set(HACK_SOURCE_LABELS)
            if unknown:
                raise ValueError(
                    f'未知数据源标签: {sorted(unknown)}，可选: {list(HACK_SOURCE_LABELS)}'
                )
            self._allowed_sources = set(sources)
        else:
            self._allowed_sources = None

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
            meta = data.get('meta') or {}
            self._facescape_root = Path(meta.get('facescape_root', facescape_root))
            self._imhead_root = Path(meta.get('imhead_root', imhead_root))
            self._ffhq_root = Path(meta.get('ffhq_root', ffhq_root))
            self._ffhq_pose_root = Path(meta.get('ffhq_pose_root', ffhq_pose_root))
            self._apply_source_filter()
            return

        self._packed_mode = False
        self._packed_items = None
        self._packed_faces = None
        fs = Path(facescape_root)
        md = Path(imhead_root)
        ffhq = Path(ffhq_root)
        self._facescape_root = fs
        self._imhead_root = md
        self._ffhq_root = ffhq
        self._ffhq_pose_root = Path(ffhq_pose_root)
        if not fs.is_dir():
            raise FileNotFoundError(f'facescape_root 不存在: {fs}')
        if not md.is_dir():
            raise FileNotFoundError(f'imhead_root 不存在: {md}')
        if not ffhq.is_dir():
            raise FileNotFoundError(f'ffhq_root 不存在: {ffhq}')
        self._obj_paths = _collect_obj_paths(fs, md, ffhq)
        if not self._obj_paths:
            raise ValueError(f'未在 {fs}、{md}、{ffhq} 下找到任何 .obj 文件')
        self._apply_source_filter()
        if not self._obj_paths:
            raise ValueError('按 sources 过滤后无可用样本')
        _, self._faces_ref_np = _read_obj_verts_tris_fast(str(self._obj_paths[0]))

    def _apply_source_filter(self) -> None:
        if self._allowed_sources is None:
            return
        if self._packed_mode:
            n_before = len(self._packed_items)
            filtered = []
            for it in self._packed_items:
                source, _ = self._resolve_source(it)
                if source in self._allowed_sources:
                    filtered.append(it)
            self._packed_items = filtered
            n_after = len(self._packed_items)
        else:
            n_before = len(self._obj_paths)
            self._obj_paths = [
                p for p in self._obj_paths
                if _source_label_for_path(
                    p,
                    self._facescape_root,
                    self._imhead_root,
                    self._ffhq_root,
                ) in self._allowed_sources
            ]
            n_after = len(self._obj_paths)
        if n_after == 0:
            raise ValueError(
                f'sources={sorted(self._allowed_sources)} 过滤后无可用样本（原 {n_before} 个）'
            )
        print(
            f'[HACKDataset] sources 过滤: {n_after}/{n_before} '
            f'({", ".join(sorted(self._allowed_sources))})',
            flush=True,
        )

    def __len__(self) -> int:
        if self._packed_mode:
            return len(self._packed_items)
        return len(self._obj_paths)  # type: ignore[arg-type]

    @staticmethod
    def _item_dict(
        verts: torch.Tensor,
        faces: torch.Tensor,
        stem: str,
        pose: Optional[torch.Tensor] = None,
        source: Optional[str] = None,
        source_id: Optional[int] = None,
        neck_pose: Optional[torch.Tensor] = None,
    ) -> dict:
        if pose is None:
            pose = torch.zeros(0, dtype=torch.float32)
        sample = {
            'verts': verts,
            'pose': pose,
            'faces': faces,
            'stem': stem,
            'source_verts': verts,
            'target_pose': pose,
        }
        if source is not None:
            sample['source'] = source
        if source_id is not None:
            sample['source_id'] = source_id
        if neck_pose is not None:
            if not isinstance(neck_pose, torch.Tensor):
                neck_pose = torch.from_numpy(np.ascontiguousarray(neck_pose))
            sample['neck_pose'] = neck_pose
        else:
            # 非 FFHQ 样本无 neck_pose；占位以保证 mixed-source batch 可 collate
            sample['neck_pose'] = torch.zeros(6, dtype=torch.float32)
        return sample

    def _resolve_neck_pose(
        self,
        item: dict,
        source: str,
        obj_path: Optional[Path] = None,
    ) -> Optional[torch.Tensor]:
        if source != SOURCE_FFHQ:
            return None
        neck_pose = item.get('neck_pose')
        if neck_pose is not None:
            if not isinstance(neck_pose, torch.Tensor):
                neck_pose = torch.from_numpy(np.ascontiguousarray(neck_pose))
            return neck_pose
        path = obj_path or item.get('obj_path')
        if path is None:
            return None
        return torch.from_numpy(
            _load_ffhq_neck_pose_for_obj(path, self._ffhq_pose_root)
        )

    def _resolve_source(self, item: dict, obj_path: Optional[Path] = None) -> Tuple[str, int]:
        source = item.get('source')
        source_id = item.get('source_id')
        if source is None and obj_path is not None:
            source = _source_label_for_path(
                obj_path,
                self._facescape_root,
                self._imhead_root,
                self._ffhq_root,
            )
        if source is None and item.get('obj_path'):
            source = _infer_source_from_obj_path(item['obj_path'])
        if source is None:
            raise ValueError(
                f'样本 {item.get("stem", "?")} 缺少 source 标签，且无法从 obj_path 推断'
            )
        if source not in HACK_SOURCE_TO_ID:
            raise ValueError(f'未知数据源标签: {source}')
        if source_id is None:
            source_id = HACK_SOURCE_TO_ID[source]
        return source, int(source_id)

    def __getitem__(self, idx: int) -> dict:
        if self._packed_mode:
            it = self._packed_items[idx]
            pose = it.get('pose')
            if pose is not None and not isinstance(pose, torch.Tensor):
                pose = torch.from_numpy(np.ascontiguousarray(pose))
            source, source_id = self._resolve_source(it)
            neck_pose = self._resolve_neck_pose(it, source)
            sample = self._item_dict(
                it['verts'],
                self._packed_faces,
                it['stem'],
                pose=pose,
                source=source,
                source_id=source_id,
                neck_pose=neck_pose,
            )
        else:
            assert self._obj_paths is not None and self._faces_ref_np is not None
            p = self._obj_paths[idx]
            v, _, stem = _load_mesh_item(p, self._faces_ref_np)
            source, source_id = self._resolve_source({}, obj_path=p)
            neck_pose = self._resolve_neck_pose({}, source, obj_path=p)
            sample = self._item_dict(
                torch.from_numpy(np.ascontiguousarray(v)),
                torch.from_numpy(np.ascontiguousarray(self._faces_ref_np)),
                stem,
                source=source,
                source_id=source_id,
                neck_pose=neck_pose,
            )

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':

    import polyscope as ps

    # if not TRAIN_HACK_PT.is_file():
        # build_hack_pt_cache(split_seed=0, num_workers=8)

    build_hack_pt_cache(split_seed=0, num_workers=8)

    dataset = HACKDataset(packed_pt=TRAIN_HACK_PT)
    print('len', len(dataset))
    s0 = dataset[100]
    print('source', s0['source'], s0['source_id'])
    if 'neck_pose' in s0:
        print('neck_pose', s0['neck_pose'].shape, s0['neck_pose'])

    print('stem', s0['stem'])
    print('verts', s0['verts'].shape, s0['verts'].dtype)
    print('faces', s0['faces'].shape, s0['faces'].dtype)

    ps.init()
    ps.register_surface_mesh(
        'HACK_mesh',
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
