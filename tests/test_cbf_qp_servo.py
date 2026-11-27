"""Offline mathematical verification, 3D kinematic stress-tests, and CLI tests for CBF-QP visual servo."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from control import servo_closed_loop as sc
from control.cbf_qp_servo import (
    CBFQPConfig,
    CBFQPVisualServoFilter,
    body_from_cv_rotation,
    extract_depth_obstacles_body,
    extract_opening_rim_obstacles_body,
    normalized_fov_jacobian,
    yaw_rotation_z,
)
from harbor import indoor_entry, part_recheck


def _project_point_normalized(
    p_world_ned: np.ndarray,
    drone_pos_ned: np.ndarray,
    drone_yaw: float,
    pitch_deg: float,
    fov_x_deg: float = 90.0,
    aspect: float = 16.0 / 9.0,
) -> tuple[float, float, float]:
    """Project a static world NED point into normalized image coords (cx, cy in [-1, 1]) and optical depth Z."""
    r_body_from_ned = yaw_rotation_z(drone_yaw).T
    r_cv_from_body = body_from_cv_rotation(pitch_deg).T
    p_body = r_body_from_ned @ (p_world_ned - drone_pos_ned)
    p_cv = r_cv_from_body @ p_body
    z = float(p_cv[2])
    tan_x = math.tan(math.radians(fov_x_deg) * 0.5)
    tan_y = tan_x / aspect
    cx = float((p_cv[0] / z) / tan_x)
    cy = float((p_cv[1] / z) / tan_y)
    return cx, cy, z


class TestCBFQPServo(unittest.TestCase):
    def test_fov_jacobian_matches_finite_difference(self) -> None:
        """Verify exact sign and magnitude of normalized_fov_jacobian against 3D pinhole finite differences."""
        for pitch_deg in (-30.0, -45.0, -90.0):
            drone_pos = np.array([10.0, -5.0, -15.0], dtype=float)
            drone_yaw = 0.65
            r_ned_from_body = yaw_rotation_z(drone_yaw)
            r_body_from_cv = body_from_cv_rotation(pitch_deg)
            p_cv_init = np.array([3.2, -1.8, 12.0], dtype=float)
            p_world = drone_pos + r_ned_from_body @ (r_body_from_cv @ p_cv_init)

            cx0, cy0, z0 = _project_point_normalized(p_world, drone_pos, drone_yaw, pitch_deg)
            j_analytic = normalized_fov_jacobian(
                cx0, cy0, z0, pitch_deg=pitch_deg, fov_x_deg=90.0, aspect_ratio=16.0 / 9.0
            )

            eps = 1e-5
            j_numeric = np.zeros((2, 4), dtype=float)
            for k in range(4):
                u = np.zeros(4, dtype=float)
                u[k] = 1.0
                pos_plus = drone_pos + r_ned_from_body @ (u[0:3] * eps)
                yaw_plus = drone_yaw + u[3] * eps
                cx_p, cy_p, _ = _project_point_normalized(p_world, pos_plus, yaw_plus, pitch_deg)

                pos_minus = drone_pos - r_ned_from_body @ (u[0:3] * eps)
                yaw_minus = drone_yaw - u[3] * eps
                cx_m, cy_m, _ = _project_point_normalized(p_world, pos_minus, yaw_minus, pitch_deg)

                j_numeric[0, k] = (cx_p - cx_m) / (2.0 * eps)
                j_numeric[1, k] = (cy_p - cy_m) / (2.0 * eps)

            err = float(np.max(np.abs(j_analytic - j_numeric)))
            self.assertLess(
                err,
                1e-4,
                f"Jacobian mismatch at pitch={pitch_deg}: max err={err}\n{j_analytic}\nvs\n{j_numeric}",
            )

    def test_depth_obstacle_extraction_rejects_outliers(self) -> None:
        """Percentile cell filter must ignore isolated near-depth spikes while detecting real obstacles."""
        depth = np.full((180, 320), 25.0, dtype=np.float32)
        depth[40, 80] = 0.4
        depth[90, 160] = 0.4
        depth[120, 240] = 0.4
        depth[75:105, 145:175] = 3.5

        pts = extract_depth_obstacles_body(depth, pitch_deg=-30.0, max_range_m=10.0, min_cell_pixels=12)
        self.assertGreaterEqual(len(pts), 1)
        dists = np.linalg.norm(pts, axis=1)
        self.assertGreater(float(dists.min()), 2.5)
        self.assertLess(float(dists.min()), 4.5)

    def test_opening_rim_extraction(self) -> None:
        """Rim extractor should sample roof-level boundary points around a circular breach."""
        h, w = 180, 320
        focal = w * 0.5
        yy, xx = np.mgrid[0:h, 0:w]
        near_d = 12.0
        r_px = 2.5 / near_d * focal
        dist_px = np.hypot(xx - (w - 1) * 0.5, yy - (h - 1) * 0.5)
        gap_mask = dist_px <= r_px
        depth = np.where(gap_mask, 22.0, near_d).astype(np.float32)

        pts = extract_opening_rim_obstacles_body(
            depth,
            visible_gap_mask=gap_mask,
            near_layer_depth_m=near_d,
            fov_x_deg=90.0,
            pitch_deg=-90.0,
            max_points=16,
        )
        self.assertGreaterEqual(len(pts), 12)
        horiz_r = np.hypot(pts[:, 0], pts[:, 1])
        med_r = float(np.median(horiz_r))
        self.assertGreaterEqual(med_r, 2.3)
        self.assertLessEqual(med_r, 2.9)

    def test_oblique_approach_fov_and_obstacle_safety(self) -> None:
        """3D kinematic stress test: large initial yaw offset + side obstacle + crosswind."""
        target_ned = np.array([26.0, 0.0, -1.5], dtype=float)
        pillar_ned = np.array([14.0, 2.2, -7.8], dtype=float)
        pillar_radius_m = 0.8
        wind_ned = np.array([0.0, 0.28, 0.0], dtype=float)

        def _run_sim(use_cbf: bool) -> dict[str, float | bool]:
            pos = np.array([0.0, 1.5, -14.5], dtype=float)
            yaw = -0.58
            pitch_deg = -30.0
            dt = 0.1
            prev_u = np.zeros(4, dtype=float)
            cbf = CBFQPVisualServoFilter(
                CBFQPConfig(
                    enable_fov_cbf=True,
                    enable_depth_cbf=True,
                    enable_obstacle_memory=True,
                    fov_max_x=0.78,
                    fov_max_y=0.78,
                    alpha_fov=1.8,
                    obs_safe_radius_m=2.2,
                    alpha_obs=1.4,
                    min_alt_m=2.0,
                    v_min=(-1.5, -1.5, -0.4, -0.55),
                    v_max=(2.2, 1.5, 1.6, 0.55),
                    max_accel_mps2=2.5,
                )
            )
            ctrl_args = type(
                "Args",
                (),
                dict(nadir=False, mass_stop=0.95, v_forward=2.2, v_down=0.85, k_vert=1.1, k_yaw=0.85, yaw_max=0.55),
            )()

            max_abs_cx = 0.0
            max_abs_cy = 0.0
            min_pillar_clearance = 999.0
            solve_times: list[float] = []
            lost_fov = False
            reached = False

            for step in range(420):
                cx, cy, z = _project_point_normalized(target_ned, pos, yaw, pitch_deg)
                if z <= 0.5 or abs(cx) > 1.0 or abs(cy) > 1.0:
                    lost_fov = True
                    break
                max_abs_cx = max(max_abs_cx, abs(cx))
                max_abs_cy = max(max_abs_cy, abs(cy))

                dist_pillar = float(np.linalg.norm(pos - pillar_ned)) - pillar_radius_m
                min_pillar_clearance = min(min_pillar_clearance, dist_pillar)

                if z <= 6.2:
                    reached = True
                    break

                f, r, d_cmd, yr, _, _ = sc.compute_body_cmd(ctrl_args, cx, cy, 0.0)
                f *= max(0.0, 1.0 - abs(cx) / 0.55)
                d_cmd = float(np.clip(d_cmd, 0.0, 1.6))
                scale = sc.depth_speed_scale(z, 18.0, 6.0, 0.38)
                f *= scale
                d_cmd *= scale

                if use_cbf:
                    r_body_from_ned = yaw_rotation_z(yaw).T
                    p_obs_body = r_body_from_ned @ (pillar_ned - pos)
                    p_dir = p_obs_body / max(float(np.linalg.norm(p_obs_body)), 1e-6)
                    obs_surface = p_obs_body - p_dir * pillar_radius_m
                    obs_pts = obs_surface[None, :] if float(np.linalg.norm(obs_surface)) < 10.0 else np.zeros((0, 3))
                    res = cbf.filter_command(
                        u_nom=(f, r, d_cmd, yr),
                        prev_u=prev_u,
                        dt=dt,
                        target_cx=cx,
                        target_cy=cy,
                        target_depth_m=z,
                        camera_pitch_deg=pitch_deg,
                        obstacles_body=obs_pts,
                        drone_pos_ned=pos,
                        drone_yaw_rad=yaw,
                        now_s=step * dt,
                        alt_m=-pos[2],
                    )
                    u_cmd = res.u_safe
                    solve_times.append(res.solve_time_ms)
                else:
                    u_cmd = np.array([f, r, d_cmd, yr], dtype=float)

                prev_u = u_cmd.copy()
                v_world = yaw_rotation_z(yaw) @ u_cmd[0:3] + wind_ned
                pos = pos + v_world * dt
                yaw = yaw + float(u_cmd[3]) * dt

            return {
                "reached": reached,
                "lost_fov": lost_fov,
                "max_abs_cx": max_abs_cx,
                "max_abs_cy": max_abs_cy,
                "min_pillar_clearance": min_pillar_clearance,
                "median_solve_ms": float(np.median(solve_times)) if solve_times else 0.0,
            }

        base = _run_sim(use_cbf=False)
        ours = _run_sim(use_cbf=True)

        self.assertTrue(ours["reached"])
        self.assertFalse(ours["lost_fov"])
        self.assertGreaterEqual(float(ours["min_pillar_clearance"]), 1.5)
        self.assertGreater(float(ours["min_pillar_clearance"]), float(base["min_pillar_clearance"]) + 0.5)
        self.assertLess(float(ours["median_solve_ms"]), 2.5)

    def test_fov_cbf_prevents_target_exit_under_aggressive_command(self) -> None:
        """Verify FOV-CBF uses exact image Jacobian to stop an aggressive forward command from pushing target out of frame."""
        pitch_deg = -30.0
        dt = 0.1
        # Target positioned near the bottom-right corner of the frame (cx~0.72, cy~0.74)
        pos0 = np.array([0.0, 0.0, -10.0], dtype=float)
        yaw0 = 0.0
        r_b_from_cv = body_from_cv_rotation(pitch_deg)
        tan_x = math.tan(math.radians(90.0) * 0.5)
        tan_y = tan_x / (16.0 / 9.0)
        z_init = 8.0
        p_cv = np.array([0.72 * tan_x * z_init, 0.74 * tan_y * z_init, z_init], dtype=float)
        target_ned = pos0 + r_b_from_cv @ p_cv

        def _run_fov_test(enable_fov: bool) -> tuple[bool, float]:
            pos = pos0.copy()
            yaw = yaw0
            prev_u = np.zeros(4, dtype=float)
            cbf = CBFQPVisualServoFilter(
                CBFQPConfig(
                    enable_fov_cbf=enable_fov,
                    enable_depth_cbf=False,
                    enable_altitude_cbf=False,
                    fov_max_x=0.78,
                    fov_max_y=0.78,
                    alpha_fov=2.0,
                    v_min=(-2.0, -2.0, -1.5, -0.6),
                    v_max=(2.5, 2.0, 2.0, 0.6),
                    max_accel_mps2=4.0,
                )
            )
            max_border = 0.0
            for _ in range(25):
                cx, cy, z = _project_point_normalized(target_ned, pos, yaw, pitch_deg)
                max_border = max(max_border, abs(cx), abs(cy))
                if abs(cx) >= 1.0 or abs(cy) >= 1.0 or z <= 0.5:
                    return True, max_border
                # Aggressive nominal forward command with weak yaw/descent correction
                u_nom = np.array([2.2, 0.0, 0.1, 0.15], dtype=float)
                res = cbf.filter_command(
                    u_nom=u_nom,
                    prev_u=prev_u,
                    dt=dt,
                    target_cx=cx,
                    target_cy=cy,
                    target_depth_m=z,
                    camera_pitch_deg=pitch_deg,
                )
                u_cmd = res.u_safe
                prev_u = u_cmd.copy()
                pos = pos + yaw_rotation_z(yaw) @ u_cmd[0:3] * dt
                yaw = yaw + float(u_cmd[3]) * dt
            return False, max_border

        lost_no_fov, max_no_fov = _run_fov_test(enable_fov=False)
        lost_with_fov, max_with_fov = _run_fov_test(enable_fov=True)
        self.assertTrue(lost_no_fov, f"Expected aggressive command without FOV-CBF to lose target, max={max_no_fov}")
        self.assertFalse(lost_with_fov, f"FOV-CBF should keep target inside frame, max={max_with_fov}")
        self.assertLess(max_with_fov, 0.88)

    def test_nadir_opening_funnel_and_blindspot_memory(self) -> None:
        """Roof-breach descent stress test with crosswind & FOV rim blind-spot."""
        roof_alt_m = 11.4
        breach_radius_m = 2.5
        uav_radius_m = 0.472
        wind_ned = np.array([0.0, 0.52, 0.0], dtype=float)

        thetas = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        rim_world = np.column_stack(
            [breach_radius_m * np.cos(thetas), breach_radius_m * np.sin(thetas), np.full_like(thetas, -roof_alt_m)]
        )

        def _run_descent(mode: str) -> dict[str, float | bool]:
            pos = np.array([0.0, 0.85, -18.0], dtype=float)
            dt = 0.1
            prev_u = np.zeros(4, dtype=float)
            cbf = CBFQPVisualServoFilter(
                CBFQPConfig(
                    enable_fov_cbf=False,
                    enable_depth_cbf=(mode != "baseline"),
                    enable_obstacle_memory=(mode == "cbf_full"),
                    enable_dob=False,
                    obs_safe_radius_m=1.25,
                    alpha_obs=1.5,
                    cylindrical_obs=True,
                    cylindrical_z_band_m=5.0,
                    min_alt_m=5.0,
                    v_min=(-1.2, -1.2, -0.5, -0.1),
                    v_max=(1.2, 1.2, 1.4, 0.1),
                    max_accel_mps2=2.5,
                )
            )
            min_rim_clearance = 999.0
            collided = False
            entered = False

            for step in range(180):
                alt = -float(pos[2])
                if roof_alt_m - 1.5 <= alt <= roof_alt_m + 0.5:
                    r_xy = math.hypot(float(pos[0]), float(pos[1]))
                    clr = (breach_radius_m - r_xy) - uav_radius_m
                    min_rim_clearance = min(min_rim_clearance, clr)
                    if clr <= 0.0:
                        collided = True
                        break
                if alt <= 6.0:
                    entered = True
                    break

                rim_visible = (alt - roof_alt_m) > 2.8
                if rim_visible:
                    u_nom = np.array([-0.25 * pos[0], -0.25 * pos[1], 1.1, 0.0], dtype=float)
                    obs_body = rim_world - pos
                else:
                    u_nom = np.array([0.0, 0.0, 1.1, 0.0], dtype=float)
                    obs_body = np.zeros((0, 3), dtype=float)

                if mode == "baseline":
                    u_cmd = u_nom
                else:
                    res = cbf.filter_command(
                        u_nom=u_nom,
                        prev_u=prev_u,
                        dt=dt,
                        obstacles_body=obs_body,
                        drone_pos_ned=pos,
                        drone_yaw_rad=0.0,
                        now_s=step * dt,
                        alt_m=alt,
                    )
                    u_cmd = res.u_safe

                prev_u = u_cmd.copy()
                pos = pos + (u_cmd[0:3] + wind_ned) * dt

            return {"entered": entered, "collided": collided, "min_rim_clearance": min_rim_clearance}

        res_base = _run_descent("baseline")
        res_nomem = _run_descent("cbf_no_memory")
        res_full = _run_descent("cbf_full")

        self.assertTrue(res_base["collided"])
        self.assertTrue(res_nomem["collided"])
        self.assertFalse(res_full["collided"])
        self.assertTrue(res_full["entered"])
        self.assertGreater(float(res_full["min_rim_clearance"]), 0.35)

    def test_dob_eliminates_crosswind_steady_state_error(self) -> None:
        """Under 0.55 m/s steady crosswind, pure P stalls at ~1.96 m while DOB-CBF-QP converges to <0.15 m."""
        wind_ned = np.array([0.0, 0.55, 0.0], dtype=float)
        dt = 0.1
        pos = np.array([0.0, 2.0, -34.0], dtype=float)
        prev_u = np.zeros(4, dtype=float)
        cbf = CBFQPVisualServoFilter(
            CBFQPConfig(
                enable_fov_cbf=False,
                enable_depth_cbf=False,
                enable_dob=True,
                v_min=(-0.8, -0.8, -0.5, -0.1),
                v_max=(0.8, 0.8, 0.5, 0.1),
                max_accel_mps2=2.5,
            )
        )
        for step in range(120):
            u_nom = np.array([-0.28 * pos[0], -0.28 * pos[1], 0.0, 0.0], dtype=float)
            res = cbf.filter_command(
                u_nom=u_nom,
                prev_u=prev_u,
                dt=dt,
                drone_pos_ned=pos,
                drone_yaw_rad=0.0,
                now_s=step * dt,
                alt_m=-pos[2],
            )
            u_cmd = res.u_safe
            prev_u = u_cmd.copy()
            pos = pos + (u_cmd[0:3] + wind_ned) * dt

        self.assertLess(abs(float(pos[1])), 0.15)
        self.assertAlmostEqual(float(cbf.dob.disturbance_ned[1]), 0.55, delta=0.08)

    def test_cli_parser_defaults_and_cbf_flags(self) -> None:
        """Ensure legacy defaults remain baseline_p and all CBF ablation flags parse cleanly."""
        args1_def = part_recheck.parse_args(["--mode", "run", "--out", "/tmp/t1"])
        self.assertEqual(args1_def.controller, "baseline_p")
        self.assertTrue(args1_def.cbf_fov)
        self.assertTrue(args1_def.cbf_depth)
        self.assertTrue(args1_def.cbf_memory)

        args1_cbf = part_recheck.parse_args(
            ["--mode", "run", "--out", "/tmp/t1", "--controller", "cbf_qp", "--no-cbf-fov", "--wind-east-mps", "0.5"]
        )
        self.assertEqual(args1_cbf.controller, "cbf_qp")
        self.assertFalse(args1_cbf.cbf_fov)
        self.assertAlmostEqual(args1_cbf.wind_east_mps, 0.5)

        args2_def = indoor_entry.parse_args(["--mode", "enter", "--out", "/tmp/t2"])
        self.assertEqual(args2_def.controller, "baseline_p")

        args2_cbf = indoor_entry.parse_args(
            ["--mode", "enter", "--out", "/tmp/t2", "--controller", "cbf_qp", "--no-cbf-memory", "--wind-north-mps", "-0.4"]
        )
        self.assertEqual(args2_cbf.controller, "cbf_qp")
        self.assertFalse(args2_cbf.cbf_memory)
        self.assertAlmostEqual(args2_cbf.wind_north_mps, -0.4)


if __name__ == "__main__":
    unittest.main()
