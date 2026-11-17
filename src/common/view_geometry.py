"""Conservative world-box visibility for sampling and absent-template labels only."""
from itertools import product
import math
import numpy as np
import cv2


def box_maybe_visible(bounds, position, rotation, fov_deg, aspect):
    corners = np.array(list(product(*zip(*bounds))), float)
    local = (corners - np.asarray(position)) @ np.asarray(rotation)
    x, y, z = local.T
    horizontal = math.tan(math.radians(fov_deg) / 2)
    vertical = horizontal / aspect
    planes = np.stack([x, horizontal*x+y, horizontal*x-y, vertical*x+z, vertical*x-z])
    return bool(np.all(planes.max(axis=1) > 0))


def visible_groups(targets, position, rotation, fov_deg, aspect):
    return {t['appearance_group'] for t in targets
            if box_maybe_visible(t['bounds_ned_m'], position, rotation, fov_deg, aspect)}


def projected_box(bounds, position, rotation, fov_deg, width, height):
    if not box_maybe_visible(bounds,position,rotation,fov_deg,width/height):
        return None
    corners = np.array(list(product(*zip(*bounds))), float)
    local = (corners-np.asarray(position)) @ np.asarray(rotation)
    if np.any(local[:,0] <= 0):
        return [0,0,width,height]
    focal = width/(2*math.tan(math.radians(fov_deg)/2))
    pixels = local[:,1:]/local[:,:1]*focal+[width/2,height/2]
    low = np.maximum(np.floor(pixels.min(0))-1,0).astype(int)
    high = np.minimum(np.ceil(pixels.max(0))+1,[width,height]).astype(int)
    return [int(low[0]),int(low[1]),int(high[0]),int(high[1])]


def redact_rgb(rgb, ignored):
    mask = cv2.resize(ignored.astype(np.uint8),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    image = rgb.copy()
    image[mask] = [124,116,104]
    return image
