import torch
import torch.nn.functional as Func


def signed_TPE_verts(verts, faces, search_tree, p=1, return_col=False):
    """
    Compute signed TPE using vertex-to-face-center distances.

    Instead of using face center pairs, this formulation measures distances from
    individual vertices of one face to the center of the colliding face.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        p (int, optional): Power for distance ratio. Defaults to 1.
        return_col (bool, optional): Whether to return number of collisions. Defaults to False.

    Returns:
        torch.Tensor: Signed TPE energy scalar
        int (optional): Number of collisions if return_col=True
    """
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        num_col = collision_idxs.shape[0]
        face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = Func.normalize(face_normals, dim=1)

    face_centers = torch.sum(triangles, dim=1) / 3 # [NF, 3]

    verts_idx = faces[collision_idxs[:, 0]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 1]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 1]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE1 = face_areas[collision_idxs[:, 1]].repeat(3) * r_x

    verts_idx = faces[collision_idxs[:, 1]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 0]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 0]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE2 = face_areas[collision_idxs[:, 0]].repeat(3) * r_x

    TPE = torch.cat([TPE1, TPE2], dim=0)
    TPE = torch.sum(TPE)

    if return_col:
        return TPE, num_col
    else:
        return TPE


def signed_TPE_verts_movable(
    verts,
    faces,
    search_tree,
    movable_faces,
    p=2,
    return_col=False,
):
    """
    Compute signed TPE only for the movable side of each collision pair.

    For a collision pair (i, j), this function adds the i -> j term only when
    face i is movable, and adds the j -> i term only when face j is movable.
    The target face center is detached in each term, so the reference side does
    not receive gradients from that term.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        movable_faces (torch.Tensor): Boolean mask of shape (F,), True for
            faces allowed to move.
        p (int, optional): Power for distance ratio. Defaults to 1.
        return_col (bool, optional): Whether to return number of collisions.
            Defaults to False.

    Returns:
        torch.Tensor: Signed TPE energy scalar
        int (optional): Number of collisions if return_col=True
    """
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        num_col = collision_idxs.shape[0]
        face_normals = torch.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = Func.normalize(face_normals, dim=1)

    if collision_idxs.shape[0] == 0:
        zero = verts.sum() * 0.0
        if return_col:
            return zero, num_col
        return zero

    movable_faces = movable_faces.to(device=faces.device, dtype=torch.bool)
    face_centers = torch.sum(triangles, dim=1) / 3  # [NF, 3]
    loss_terms = []

    collision_faces = torch.unique(collision_idxs.reshape(-1))
    movable_collision_faces = collision_faces[movable_faces[collision_faces]]

    if movable_collision_faces.shape[0] > 0:
        movable_collision_mask = torch.zeros_like(movable_faces)
        movable_collision_mask[movable_collision_faces] = True

        moving_faces = torch.cat(
            [collision_idxs[:, 0], collision_idxs[:, 1]],
            dim=0,
        )
        reference_faces = torch.cat(
            [collision_idxs[:, 1], collision_idxs[:, 0]],
            dim=0,
        )
        active = movable_collision_mask[moving_faces]
        moving_faces = moving_faces[active]
        reference_faces = reference_faces[active]

        verts_idx = faces[moving_faces]  # [Cm, 3]
        X = verts[verts_idx].reshape(-1, 3)  # [3Cm, 3]
        Y = face_centers[reference_faces].detach().repeat_interleave(3, dim=0)

        dist = torch.linalg.vector_norm(Y - X, dim=1)
        Pdist_x = torch.sum(
            (Y - X) * face_normals[reference_faces].repeat_interleave(3, dim=0),
            dim=1,
        )
        r_x = (Pdist_x / dist).pow(p)
        loss_terms.append(face_areas[reference_faces].repeat_interleave(3) * r_x)

    if loss_terms:
        TPE = torch.sum(torch.cat(loss_terms, dim=0))
    else:
        TPE = verts.sum() * 0.0

    if return_col:
        return TPE, num_col
    else:
        return TPE


def signed_TPE_verts_mask(verts, faces, search_tree, mask1, mask2, p=3, return_col=False):
    """
    Compute signed TPE with masked face normals.

    This variant allows masking certain faces to use custom normals instead of
    computed face normals, useful for handling special geometric configurations.

    Args:
        verts (torch.Tensor): Vertex positions of shape (V, 3)
        faces (torch.Tensor): Face indices of shape (F, 3)
        search_tree (BVH): BVH search tree for collision detection
        mask1 (torch.Tensor): Boolean mask for first set of faces
        mask2 (torch.Tensor): Boolean mask for second set of faces
        p (int, optional): Power for distance ratio. Defaults to 3.
        return_col (bool, optional): Whether to return number of collisions. Defaults to False.

    Returns:
        torch.Tensor: Signed TPE energy scalar
        int (optional): Number of collisions if return_col=True
    """
    triangles = verts[faces]
    with torch.no_grad():
        collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
        collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]
        #  print(collision_idxs.shape)
        num_col = collision_idxs.shape[0]
        face_normals = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * torch.norm(face_normals, dim=1)
        face_normals = Func.normalize(face_normals, dim=1)
        face_normals[mask1] = torch.zeros_like(face_normals[mask1])
        face_normals[mask1, 2] = 1.0
        face_normals[mask2] = torch.ones_like(face_normals[mask2])
        face_normals[mask2, 2] = 1.0

    face_centers = torch.sum(triangles, dim=1) / 3 # [NF, 3]

    verts_idx = faces[collision_idxs[:, 0]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 1]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 1]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE1 = face_areas[collision_idxs[:, 1]].repeat(3) * r_x

    verts_idx = faces[collision_idxs[:, 1]] # [C, 3]
    X = verts[verts_idx].reshape(-1, 3) # [3C, 3]
    Y = face_centers[collision_idxs[:, 0]].repeat(3, 1)

    dist = torch.linalg.vector_norm(Y - X, dim=1)
    Pdist_x = torch.sum((Y - X) * face_normals[collision_idxs[:, 0]].repeat(3, 1), dim=1)
    r_x = (Pdist_x / dist).pow(p)
    #  r_x = (Pdist_x ).pow(p)
    TPE2 = face_areas[collision_idxs[:, 0]].repeat(3) * r_x

    TPE = torch.cat([TPE1, TPE2], dim=0)
    TPE = torch.sum(TPE)

    if return_col:
        return TPE, num_col
    else:
        return TPE
