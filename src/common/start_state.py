"""Compare an experimental start with its shared pose and stationary limits."""
import math
import numpy as np


def start_state_check(kinematics, expected):
    pose = kinematics['pose']
    position = np.array([pose['position'][k] for k in ('x', 'y', 'z')])
    q = pose['orientation']
    yaw = math.atan2(2*(q['w']*q['z']+q['x']*q['y']),
                     1-2*(q['y']**2+q['z']**2))
    delta = yaw-math.radians(expected['yaw_deg'])
    velocity = kinematics.get('twist', {}).get('linear', kinematics.get('linear_velocity'))
    if velocity is None:
        raise ValueError('Kinematics must include measured linear velocity')
    result = {
        'position_ned_m': position.tolist(),
        'yaw_deg': math.degrees(yaw),
        'position_error_m': float(np.linalg.norm(position-expected['position_ned_m'])),
        'yaw_error_deg': abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta)))),
        'speed_mps': float(np.linalg.norm([velocity[k] for k in ('x', 'y', 'z')])),
        'roll_pitch_deg': [math.degrees(math.atan2(2*(q['w']*q['x']+q['y']*q['z']),
                                                  1-2*(q['x']**2+q['y']**2))),
                           math.degrees(math.asin(float(np.clip(2*(q['w']*q['y']-q['z']*q['x']), -1, 1))))],
    }
    result['passed'] = (result['position_error_m'] <= expected['position_tolerance_m']
                        and result['yaw_error_deg'] <= expected['yaw_tolerance_deg']
                        and result['speed_mps'] <= expected['speed_tolerance_mps']
                        and max(map(abs, result['roll_pitch_deg'])) <= expected['tilt_tolerance_deg'])
    return result
