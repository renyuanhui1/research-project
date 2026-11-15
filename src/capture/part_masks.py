"""Raycast the target mesh through the fixed camera to obtain per-part masks."""
import cv2
import numpy as np
import open3d as o3d


def part_membership(points, part, origin, forward, reference):
    """Aircraft-scene bounds from the scene profile, same rule as evaluation labels."""
    points = np.atleast_2d(np.asarray(points, float))
    origin = np.asarray(origin, float)
    forward = np.asarray(forward, float)
    lateral = np.array([-forward[1], forward[0]])
    longitudinal = (points[:, :2]-origin[:2])@forward
    side = (points[:, :2]-origin[:2])@lateral
    valid = (np.linalg.norm(points-np.asarray(reference, float), axis=1) <= part['radius_m'])
    valid &= longitudinal > part.get('longitudinal_min_m', -np.inf)
    valid &= longitudinal < part.get('longitudinal_max_m', np.inf)
    valid &= side >= part.get('lateral_min_m', -np.inf)
    valid &= side <= part.get('lateral_max_m', np.inf)
    valid &= -points[:, 2] > part.get('min_altitude_m', -np.inf)
    valid &= -points[:, 2] < part.get('max_altitude_m', np.inf)
    return valid


def load_mesh(path):
    with np.load(path) as data:
        vertices = data['vertices_enu'][:, [1, 0, 2]].astype(np.float32)
        vertices[:, 2] *= -1
        faces = data['triangles'].astype(np.uint32)
    if not len(vertices) or not len(faces):
        raise ValueError('Empty target mesh: '+str(path))
    return vertices, faces


class PartMaskRenderer:
    """One raycasting scene and one camera ray grid; rotate rays per frame."""

    def __init__(self, mesh_path, parts, origin, forward, reference,
                 camera_offset, camera_mount, shape=(1080, 1920),
                 horizontal_fov_deg=90., stride=2):
        self.vertices, faces = load_mesh(mesh_path)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.vertices), o3d.core.Tensor(faces))
        self.parts = parts
        self.origin = np.asarray(origin, float)
        self.forward = np.asarray(forward, float)
        self.reference = np.asarray(reference, float)
        self.offset = np.asarray(camera_offset, float)
        self.mount = np.asarray(camera_mount, float)
        h, w = shape
        if not 0 < stride <= 16 or horizontal_fov_deg <= 0:
            raise ValueError('Invalid mask sampling')
        us, vs = np.arange(0, w, stride), np.arange(0, h, stride)
        uu, vv = np.meshgrid(us, vs)
        focal = w/(2*np.tan(np.radians(horizontal_fov_deg)/2))
        directions = np.stack([np.ones(uu.shape), (uu-w/2)/focal, (vv-h/2)/focal], -1)
        self.directions = directions.reshape(-1, 3).astype(np.float32)
        self.grid = (len(vs), len(us))
        self.shape = (h, w)

    def render(self, body_position, body_rotation):
        position = np.asarray(body_position, float)+np.asarray(body_rotation)@self.offset
        rotation = np.asarray(body_rotation)@self.mount
        return self.render_camera(position, rotation)

    def render_camera(self, position, rotation):
        """Use the recorded camera pose; required when the vehicle moves between the image and its kinematics."""
        position = np.asarray(position, float)
        rotation = np.asarray(rotation, float)
        directions = self.directions@rotation.T
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        rays = np.concatenate([np.broadcast_to(position, directions.shape), directions], 1)
        distance = self.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
        hit = np.isfinite(distance) & (distance > 0)
        points = position+np.where(hit, distance, 0.)[:, None]*directions
        masks = {'aircraft': hit}
        for name, part in self.parts.items():
            masks[name] = hit & part_membership(points, part, self.origin, self.forward, self.reference)
        h, w = self.shape
        return {name: cv2.resize(mask.reshape(self.grid).astype(np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST)
                for name, mask in masks.items()}
