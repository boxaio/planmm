import numpy as np
import torch
import polyscope as ps
import trimesh
import polyscope.imgui as psim
from typing import List, Literal, Optional, Tuple


# def read_obj(path):
#     mesh = openmesh.read_trimesh(path)
#     pos = torch.from_numpy(mesh.points()).to(torch.float)
#     face = torch.from_numpy(mesh.face_vertex_indices())
#     face = face.t().to(torch.long).contiguous()
#     return Data(pos=pos, face=face)

def hex_to_rgb(hex_color):
    hex_color = hex_color.strip()
    if hex_color.startswith('#'):
        hex_str = hex_color[1:]
    else:
        hex_str = hex_color
    
    if len(hex_str) != 6:
        raise ValueError("无效的十六进制颜色码，需为6位（含#则7位），例如 #58C4DD")
    
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
    except ValueError:
        raise ValueError("颜色码包含非十六进制字符（仅支持 0-9, a-f, A-F）")
    
    return (r, g, b)


def show_pointcloud(points: torch.Tensor, radius: float=0.004):
    assert len(points.shape) == 2 and points.shape[-1] == 3
    # range scale of the points

    ps.init()
    ps.register_point_cloud(
        "cloud", 
        points=points, 
        radius=radius,
        color=np.array(hex_to_rgb('#58C4DD'))/255.0,
    )
    ps.show()


def show_mesh(verts: torch.Tensor, faces: torch.Tensor, show_edge: bool=True, smooth_shade: bool=True):
    assert len(verts.shape) == 2 and verts.shape[-1] == 3
    assert len(faces.shape) == 2 and faces.shape[-1] == 3

    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    ps.init()
    ps.register_surface_mesh(
        "mesh", 
        vertices=verts, 
        faces=faces,
        enabled=True,
        color=(0.75, 0.75, 0.75),
        edge_width=0.9 if show_edge else 0,
        material='clay',
        smooth_shade=smooth_shade,
        back_face_policy='custom',
    )
    ps.show()


def show_mesh_pair(
    verts_a: torch.Tensor,
    faces_a: torch.Tensor,
    verts_b: torch.Tensor,
    faces_b: torch.Tensor,
    *,
    show_edge: bool = True,
    smooth_shade: bool = True,
    gap: Optional[float] = None,
    use_offset: bool = True,
    colors: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.72, 0.72, 0.78),
        (0.78, 0.72, 0.68),
    ),
    names: Tuple[str, str] = ("mesh_a", "mesh_b"),
):
    """并排显示两个三角网格：将 ``verts_b`` 沿 +x 平移，使其左边界与 ``verts_a`` 右边界相距 ``gap``。

    ``gap`` 为 ``None`` 时取两边包围盒对角线长度的约 5%，避免贴得过紧。
    """
    assert len(verts_a.shape) == 2 and verts_a.shape[-1] == 3
    assert len(faces_a.shape) == 2 and faces_a.shape[-1] == 3
    assert len(verts_b.shape) == 2 and verts_b.shape[-1] == 3
    assert len(faces_b.shape) == 2 and faces_b.shape[-1] == 3

    va, fa = verts_a, faces_a
    vb, fb = verts_b, faces_b
    if isinstance(va, torch.Tensor):
        va = va.detach().cpu().numpy()
    if isinstance(fa, torch.Tensor):
        fa = fa.detach().cpu().numpy()
    if isinstance(vb, torch.Tensor):
        vb = vb.detach().cpu().numpy()
    if isinstance(fb, torch.Tensor):
        fb = fb.detach().cpu().numpy()

    va = np.asarray(va, dtype=np.float64)
    vb = np.asarray(vb, dtype=np.float64)

    ba_min, ba_max = va.min(axis=0), va.max(axis=0)
    bb_min, bb_max = vb.min(axis=0), vb.max(axis=0)
    combined_diag = np.linalg.norm(np.maximum(ba_max, bb_max) - np.minimum(ba_min, bb_min))
    if gap is None:
        gap = float(max(combined_diag * 0.05, 1e-6))

    xa_max = float(va[:, 0].max())
    xb_min = float(vb[:, 0].min())
    if use_offset:
        offset_x = (xa_max + gap) - xb_min
        vb_shifted = vb.copy()
        vb_shifted[:, 0] += offset_x
    else:
        vb_shifted = vb.copy()


    ps.init()
    ps.register_surface_mesh(
        names[0],
        vertices=va,
        faces=fa,
        enabled=True,
        color=colors[0],
        edge_width=0.9 if show_edge else 0,
        material="clay",
        smooth_shade=smooth_shade,
        back_face_policy="custom",
    )
    ps.register_surface_mesh(
        names[1],
        vertices=vb_shifted,
        faces=fb,
        enabled=True,
        color=colors[1],
        edge_width=0.9 if show_edge else 0,
        material="clay",
        smooth_shade=smooth_shade,
        back_face_policy="custom",
    )
    ps.show()


def show_pointcloud_scalar_field(points: torch.Tensor, scalar_field: torch.Tensor, radius: float=0.004):
    assert len(points.shape) == 2 and points.shape[-1] == 3
    assert len(scalar_field.shape) == 1 and scalar_field.shape[0] == points.shape[0]

    ps.init()
    ps_cloud = ps.register_point_cloud(
        "cloud", 
        points=points, 
        radius=radius,
        color=np.array(hex_to_rgb('#58C4DD'))/255.0,
    )
    ps_cloud.add_scalar_quantity(
        "scalar_field", 
        scalar_field, 
        cmap='plasma',
        enabled=True,
    )
    ps.show()

def show_pointcloud_vector_field(points: torch.Tensor, vector_field: torch.Tensor, radius: float=0.004):
    assert len(points.shape) == 2 and points.shape[-1] == 3
    assert len(vector_field.shape) == 2 and vector_field.shape[0] == points.shape[0]

    ps.init()
    ps_cloud = ps.register_point_cloud(
        "cloud", 
        points=points, 
        radius=radius,
        color=np.array(hex_to_rgb('#58C4DD'))/255.0,
    )
    ps_cloud.add_vector_quantity(
        "vector_field", 
        vector_field, 
        radius=0.001, 
        length=0.05, 
        color=np.array(hex_to_rgb('#FC6255'))/255.0,
    )
    ps.show()


def show_mesh_scalar_field(
    verts: torch.Tensor, faces: torch.Tensor, scalar_field: torch.Tensor, 
    defined_on: Literal['vertices', 'faces', 'edges'], 
    show_edge: bool=True, smooth_shade: bool=True,
):
    assert len(verts.shape) == 2 and verts.shape[-1] == 3
    assert len(faces.shape) == 2 and faces.shape[-1] == 3
    assert len(scalar_field.shape) == 1

    ps.init()
    ps_mesh = ps.register_surface_mesh(
        "mesh", 
        vertices=verts, 
        faces=faces,
        enabled=True,
        color=(0.75, 0.75, 0.75),
        edge_width=0.9 if show_edge else 0,
        material='clay',
        smooth_shade=smooth_shade,
    )
    if defined_on == 'vertices':
        assert scalar_field.shape[0] == verts.shape[0]
    elif defined_on == 'faces':
        assert scalar_field.shape[0] == faces.shape[0]
    elif defined_on == 'edges':
        assert scalar_field.shape[0] == ps_mesh.n_edges()

    ps_mesh.add_scalar_quantity(
        "scalar_field", 
        scalar_field, 
        defined_on=defined_on,
        cmap='reds',
        enabled=True,
    )
    ps.show()

def show_mesh_vector_field(
    verts: torch.Tensor, faces: torch.Tensor, vector_field: torch.Tensor, 
    defined_on: Literal['vertices', 'faces'], 
    show_edge: bool=True, smooth_shade: bool=True,
    radius: float=0.001,
    length: float=0.05,
):
    assert len(verts.shape) == 2 and verts.shape[-1] == 3
    assert len(faces.shape) == 2 and faces.shape[-1] == 3
    assert len(vector_field.shape) == 2 and vector_field.shape[-1] == 3

    ps.init()
    ps_mesh = ps.register_surface_mesh(
        "mesh", 
        vertices=verts, 
        faces=faces,
        enabled=True,
        color=(0.75, 0.75, 0.75),
        edge_width=0.9 if show_edge else 0,
        material='clay',
        smooth_shade=smooth_shade,
    )
    if defined_on == 'vertices':
        assert vector_field.shape[0] == verts.shape[0]
    elif defined_on == 'faces':
        assert vector_field.shape[0] == faces.shape[0]

    ps_mesh.add_vector_quantity(
        "vector_field", 
        vector_field, 
        defined_on=defined_on,
        enabled=True,
        radius=radius, 
        length=length, 
        color=np.array(hex_to_rgb('#FC6255'))/255.0,
    )
    ps.show()


def _numpy_xyz(
    verts: torch.Tensor | np.ndarray,
    faces: torch.Tensor | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    return verts, faces


def _interleaved_face_grad_to_per_face_scalar(
    g2f: np.ndarray,
    n_faces: int,
    agg: str,
) -> np.ndarray:
    """``g2f``: (2*F, 3) intrinsic gradient rows interleaved per face (与 ``G`` / ``M`` 约定一致)."""
    if g2f.shape != (2 * n_faces, 3):
        raise ValueError(
            f"preds_grad 期望形状 (2*F, 3) = ({2 * n_faces}, 3)，实际 {g2f.shape}"
        )
    ga = g2f[0::2]
    gb = g2f[1::2]
    na = np.linalg.norm(ga, axis=-1)
    nb = np.linalg.norm(gb, axis=-1)
    if agg == "mean_norm":
        return 0.5 * (na + nb)
    if agg == "rms_norm":
        return np.sqrt(0.5 * (na * na + nb * nb))
    if agg == "max_norm":
        return np.maximum(na, nb)
    if agg == "l2_fro":
        return np.sqrt(np.sum(ga**2 + gb**2, axis=-1))
    raise ValueError(
        f"未知 agg={agg!r}，可选: 'mean_norm', 'rms_norm', 'max_norm', 'l2_fro'"
    )


def show_mesh_grad_cloud(
    verts: torch.Tensor | np.ndarray,
    faces: torch.Tensor | np.ndarray,
    grad: torch.Tensor | np.ndarray,
    *,
    batch_index: int = 0,
    agg: Literal["mean_norm", "rms_norm", "max_norm", "l2_fro"] = "mean_norm",
    mesh_name: str = "mesh",
    quantity_name: str = "pred_grad",
    cmap: str = "reds",
    show_edge: bool = True,
    smooth_shade: bool = False,
) -> None:
    """在三角网格上用伪彩色（云图）显示 ``grad``。

    ``grad`` 与 Poisson 分支一致，形状为 ``[B, 2*F, 3]`` 或 ``[2*F, 3]``：
    每个面两条交错行（内在梯度的两个分量），``F`` 须与 ``faces.shape[0]`` 一致。

    ``agg`` 将每个面的两行 3D 向量聚合成一个标量供面片着色。
    """
    v, f = _numpy_xyz(verts, faces)
    if isinstance(grad, torch.Tensor):
        grad = grad.detach().cpu().numpy()
    g = np.asarray(grad, dtype=np.float64)
    if g.ndim == 3:
        g = g[int(batch_index)]
    if g.ndim != 2 or g.shape[-1] != 3:
        raise ValueError(f"grad 应为 (2*F, 3) 或 (B, 2*F, 3)，当前 {g.shape}")

    n_face = int(f.shape[0])
    face_scalar = _interleaved_face_grad_to_per_face_scalar(g, n_face, agg)

    ps.init()
    ps_mesh = ps.register_surface_mesh(
        mesh_name,
        vertices=v,
        faces=f,
        enabled=True,
        color=(0.75, 0.75, 0.75),
        edge_width=0.9 if show_edge else 0,
        material="clay",
        smooth_shade=smooth_shade,
        back_face_policy="custom",
    )
    ps_mesh.add_scalar_quantity(
        quantity_name,
        face_scalar,
        defined_on="faces",
        cmap=cmap,
        enabled=True,
    )
    ps.show()


def show_mesh_knn_gui(
    verts: torch.Tensor, faces: torch.Tensor, k: int, edge_index: torch.Tensor, 
    show_edge: bool=True, smooth_shade: bool=True,
):
    assert len(verts.shape) == 2 and verts.shape[-1] == 3
    assert len(faces.shape) == 2 and faces.shape[-1] == 3

    from models.meshes.common import knn_graph_mesh
    
    select_type = 'vertices'
    selected_vertices = [0]
    last_selected_vertex = None
    default_color = np.array([0.75, 0.75, 0.75])

    ps.init()
    ps_mesh = ps.register_surface_mesh(
        "mesh", 
        vertices=verts.numpy(), 
        faces=faces.numpy(),
        enabled=True,
        color=default_color,
        edge_width=0.9 if show_edge else 0,
        material='clay',
        smooth_shade=smooth_shade,
    )

    def callback():
        # global verts, faces, edge_index
        global select_type, selected_vertices, last_selected_vertex, vertex_color

        psim.TextUnformatted("Interactive Mesh KNN")
        psim.Separator()

        # get current user selected vertex
        select_type, selected_vertex = get_user_selection(verts, faces)
        selected_vertices = [selected_vertex]
        row, col = edge_index
        
        if len(selected_vertices) == 1:
            vertex_color = np.tile(np.array([0.75, 0.75, 0.75]), (verts.shape[0], 1))
            k_ids = col[row==selected_vertex]
            vertex_color[k_ids] = np.tile(np.array(hex_to_rgb('#FC6255'))/255.0, (len(k_ids), 1))
            ps.get_surface_mesh("mesh").add_color_quantity("show_color", vertex_color, enabled=True)
            ps.register_point_cloud(
                "point", 
                points=verts[k_ids].numpy(), 
                radius=0.001,
                # color=np.array(hex_to_rgb('#58C4DD'))/255.0,
                enabled=True,
            )
            point_color = np.tile(np.array([0.95, 0.05, 0.05]), (verts[k_ids].shape[0], 1))
            point_color[0] = np.array([0.1, 0.9, 0.9])
            ps.get_point_cloud("point").add_color_quantity("point_color", point_color, enabled=True)

        selected_vertices_str = '[' + ' '.join(['%d' % idx for idx in selected_vertices]) + ']'
        psim.TextUnformatted(f"Selected vertices: {selected_vertices_str}")

        if(psim.Button("Undo")):
            # This code is executed when the button is pressed      
            if len(selected_vertices) > 0:
                last_selected_vertex = selected_vertices[-1]
                selected_vertices = selected_vertices[:-1]
        psim.SameLine()

        if(psim.Button("Redo")):
            # This code is executed when the button is pressed
            not_first_negative = len(selected_vertices) == 0 and last_selected_vertex > 0
            different_from_last = len(selected_vertices) > 0 and last_selected_vertex != selected_vertices[-1]

            if last_selected_vertex is not None:
                if not_first_negative or different_from_last:
                    selected_vertices.append(last_selected_vertex)
        psim.SameLine()

        if(psim.Button("Reset")):
            # This code is executed when the button is pressed
            if len(selected_vertices) > 0:
                last_selected_vertex = selected_vertices[-1]    
            
            selected_vertices = []

    ps.set_user_callback(callback)

    ps.show()



if __name__ == '__main__':
    
    mesh = trimesh.load('/home/ubuntu/MyExp/PHACK/figures/Hallo3_000176_60_0.obj')
    verts = torch.tensor(mesh.vertices).float()
    faces = torch.tensor(mesh.faces).long()

    from models.meshes.common import knn_graph_mesh

    k = 20
    edge_index = knn_graph_mesh(verts, k)

    show_mesh_knn_gui(verts, faces, k, edge_index=edge_index)


    # from models.meshes.common import knn_graph_mesh

    # edge_index = knn_graph_mesh(verts, k=10)
    # row, col = edge_index

    # vert_id = 10
    # print(col[row == vert_id]) 

    

