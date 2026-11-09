"""Height colors shared by geometry views."""
from std_msgs.msg import ColorRGBA

def height_color(t):
    """低处蓝、高处红的简易 jet 色表。"""
    t = min(max(float(t), 0.0), 1.0)
    r = min(max(1.5 - abs(4 * t - 3), 0.0), 1.0)
    g = min(max(1.5 - abs(4 * t - 2), 0.0), 1.0)
    b = min(max(1.5 - abs(4 * t - 1), 0.0), 1.0)
    return ColorRGBA(r=r, g=g, b=b, a=0.92)
