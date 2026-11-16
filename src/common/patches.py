"""Read RGB patches and part targets from HDF5 or an existing image directory."""
from contextlib import nullcontext
from pathlib import Path

import cv2
import h5py
import numpy as np


def open_patches(path, mode='r'):
    path = Path(path)
    return nullcontext(path) if path.is_dir() else h5py.File(path, mode)


def has_patch(store, sample):
    return (store/(sample+'.png')).is_file() if isinstance(store, Path) else sample in store


def read_rgb(store, sample):
    if isinstance(store, Path):
        image = cv2.imread(str(store/(sample+'.png')))
    else:
        image = cv2.imdecode(store[sample]['rgb_png'][()], cv2.IMREAD_COLOR)
    return np.ascontiguousarray(image[:, :, ::-1])


def read_target(store, sample, part):
    if isinstance(store, Path):
        path = store/(sample+'-'+part+'.npy')
        return np.load(path if path.exists() else store/(sample+'-mask.npy'))
    return store[sample]['targets'][part][()]


def write_targets(store, sample, targets):
    if isinstance(store, Path):
        for part, target in targets.items():
            np.save(store/(sample+'-'+part+'.npy'), target)
        return
    group = store.require_group(sample).require_group('targets')
    for part, target in targets.items():
        if part in group:
            group[part][...] = target
        else:
            group.create_dataset(part, data=target, compression='gzip', compression_opts=1)


def write_sample(store, sample, rgb, targets):
    encoded = cv2.imencode('.png', rgb[:, :, ::-1])[1]
    store.create_group(sample).create_dataset('rgb_png', data=encoded)
    write_targets(store, sample, targets)
