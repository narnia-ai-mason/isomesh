"""Manifold/watertight mesh validation utilities."""

from collections import defaultdict
import numpy as np


def check_manifold_watertight(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Comprehensive manifold/watertight validation.

    Returns a dict with:
      - is_manifold: bool (every edge shared by exactly 2 faces)
      - is_watertight: bool (no boundary edges)
      - euler_characteristic: int
      - non_manifold_edges: list
      - boundary_edges: list
      - consistent_winding: bool
      - V, E, F: counts
    """
    edge_faces = defaultdict(list)
    directed_edge_faces = defaultdict(list)

    for fi, face in enumerate(faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for v0, v1 in [(a, b), (b, c), (c, a)]:
            edge_key = (min(v0, v1), max(v0, v1))
            edge_faces[edge_key].append(fi)
            directed_edge_faces[(v0, v1)].append(fi)

    V = len(vertices)
    E = len(edge_faces)
    F = len(faces)
    euler = V - E + F

    non_manifold_edges = [e for e, fs in edge_faces.items() if len(fs) != 2]
    boundary_edges = [e for e, fs in edge_faces.items() if len(fs) == 1]

    # Consistent winding: for each undirected edge shared by 2 faces,
    # the two directed half-edges should appear exactly once each
    consistent_winding = True
    for (v0, v1), fs in edge_faces.items():
        if len(fs) == 2:
            count_01 = len(directed_edge_faces.get((v0, v1), []))
            count_10 = len(directed_edge_faces.get((v1, v0), []))
            if not (count_01 == 1 and count_10 == 1):
                consistent_winding = False
                break

    return {
        "is_manifold": len(non_manifold_edges) == 0,
        "is_watertight": len(boundary_edges) == 0,
        "euler_characteristic": euler,
        "non_manifold_edges": non_manifold_edges[:10],
        "boundary_edges": boundary_edges[:10],
        "consistent_winding": consistent_winding,
        "V": V,
        "E": E,
        "F": F,
    }


def check_no_degenerate_faces(faces: np.ndarray) -> bool:
    """Check that no face has repeated vertex indices."""
    for face in faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        if a == b or b == c or a == c:
            return False
    return True


def check_face_indices_valid(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Check that all face indices reference valid vertices."""
    if len(faces) == 0:
        return True
    return int(faces.min()) >= 0 and int(faces.max()) < len(vertices)
