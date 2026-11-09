"""RGB、深度与分割消息解码，以及场景分割ID设置。"""

import numpy as np

from common.profiles import selected_profile
PROFILE = selected_profile()
DEFAULT_TARGET = PROFILE["target"]["actor"]
SEG_IDS = PROFILE["target"]["segmentation_ids"]

def decode(msg):
    """相机消息 → ndarray[h,w,c]。深度可能是 16UC1 / 32FC1 / 16FC1，都要认。"""
    h, w = msg["height"], msg["width"]
    enc = str(msg["encoding"]).upper()
    specs = {"BGR": (np.uint8, 3), "RGB": (np.uint8, 3), "BGRA": (np.uint8, 4),
             "RGBA": (np.uint8, 4), "8UC1": (np.uint8, 1), "16UC1": (np.uint16, 1),
             "32FC1": (np.float32, 1), "16FC1": (np.float16, 1)}
    if enc not in specs:
        raise ValueError(f"不认识的 encoding={enc!r}")
    dt, ch = specs[enc]
    data = msg["data"]
    buf = (np.frombuffer(data, dtype=dt) if isinstance(data, (bytes, bytearray, memoryview))
           else np.asarray(data, dtype=dt))
    if buf.size != h * w * ch:
        raise ValueError(f"元素数 {buf.size} != 预期 {h*w*ch}")
    return buf.reshape(h, w, ch)

def decode_image(image_msg):
    """把一条相机 image_msg 解成标准 RGB 的 numpy 数组 (H, W, 3) uint8。

    - 通道数 C 用 len(data)/(H*W) 反推校验，不硬编码。
    - 按 encoding 决定通道顺序，统一转成标准 RGB。
    返回 (rgb, info)；info 含 encoding/channels 便于打印确认。
    """
    h, w = image_msg["height"], image_msg["width"]
    data = image_msg["data"]
    enc = str(image_msg["encoding"]).lower()

    # Pub/Sub 经 msgpack 传来的是 bytes/buffer；reqrep 经 JSON 传来的是 list。
    if isinstance(data, list):
        arr = np.array(data, dtype=np.uint8)
    else:
        arr = np.frombuffer(data, dtype=np.uint8)

    n = arr.size
    pixels = h * w
    if pixels == 0 or n % pixels != 0:
        raise ValueError(f"len(data)={n} 无法被 H*W={pixels} 整除，无法反推通道数")
    channels = n // pixels

    img = arr.reshape(h, w, channels)

    # 统一转成标准 RGB
    if channels == 3:
        rgb = img[..., ::-1] if enc.startswith("bgr") else img
    elif channels == 4:
        # bgra/rgba：先取前三通道，再按需翻转
        rgb = img[..., 2::-1] if enc.startswith("bgr") else img[..., :3]
    elif channels == 1:
        rgb = np.repeat(img, 3, axis=2)
    else:
        raise ValueError(f"未预期的通道数 channels={channels} (encoding={enc})")

    return np.ascontiguousarray(rgb), {"encoding": enc, "channels": channels}

def setup_segmentation(world, names=None):
    """运行时覆盖分割 ID（**每次载入场景都要重做**，不写回配置）。

    默认 ID 全撞 83：哈希 seg_id = 5 + Σ(ASCII≥97 的字符)，数字和下划线被跳过，
    故 staticmeshactor_22 与 _14 参与哈希的字符完全相同。覆盖走字符串精确比较，绕开哈希。
    names 为空时使用场景配置中的映射；也可只覆盖当前目标。
    """
    selected = SEG_IDS if names is None else {name: SEG_IDS[name] for name in names}
    ok = 0
    for name, sid in selected.items():
        try:
            if world.set_segmentation_id_by_name(name, sid, False, True):
                ok += 1
        except Exception as err:
            print(f"  分割 ID 设置失败 {name}: {err}")
    print(f"分割 ID 已覆盖 {ok}/{len(selected)}")
