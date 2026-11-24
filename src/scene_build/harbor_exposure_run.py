"""Measure baked-in level exposure along a smooth camera approach to the fish-market wall."""

import json
import math
import time
import traceback
from pathlib import Path
import unreal as u

work = Path(__file__).parent
out = work / "approach"
out.mkdir(exist_ok=True)
station = json.loads((work / "market-stations.json").read_text())["Building_071"]
aim = u.Vector(*(v * 100 for v in station["aim"]))
normal = station["front_normal"]
world = u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
actors = u.get_editor_subsystem(u.EditorActorSubsystem)
time0 = time.monotonic()
camera = None
last_d = None
samples = iter([65, 55, 45, 35, 25, 14])
threshold = next(samples)
records = []


def move(distance):
    eye = aim + u.Vector(
        normal[0] * distance * 100, normal[1] * distance * 100, distance * 43
    )
    r = u.MathLibrary.find_look_at_rotation(eye, aim)
    camera.set_actor_location_and_rotation(eye, r, False, False)
    u.EditorLevelLibrary.set_level_viewport_camera_info(eye, r)


def tick(dt):
    global camera, cap, rt, time0, last_d, threshold
    try:
        now = time.monotonic()
        if camera is None:
            if now - time0 < 3:
                return
            u.AutomationLibrary.finish_loading_before_screenshot()
            u.SystemLibrary.execute_console_command(
                world, "r.Shadow.Virtual.MaxPhysicalPagesSceneCapture 1536"
            )
            camera = actors.spawn_actor_from_class(u.SceneCapture2D, aim)
            cap = camera.get_component_by_class(u.SceneCaptureComponent2D)
            rt = u.RenderingLibrary.create_render_target2d(
                world, 1200, 750, u.TextureRenderTargetFormat.RTF_RGBA8
            )
            rt.set_editor_property("target_gamma", 2.2)
            for k, v in dict(
                texture_target=rt,
                capture_source=u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,
                fov_angle=68,
                capture_every_frame=True,
                always_persist_rendering_state=True,
                post_process_blend_weight=1.0,
            ).items():
                cap.set_editor_property(k, v)
            move(65)
            last_d = 65
            time0 = time.monotonic()
            return
        elapsed = now - time0
        if elapsed < 6:
            return
        if last_d <= threshold:
            name = f"wall_{threshold:02d}"
            u.RenderingLibrary.export_render_target(world, rt, str(out), name + ".png")
            eye = camera.get_actor_location()
            hit = u.SystemLibrary.line_trace_single(
                world,
                eye,
                aim + (aim - eye) * 0.22,
                u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                True,
                [camera],
                u.DrawDebugTrace.NONE,
                True,
            )
            actor_hits = (
                []
                if not hit
                else [
                    v.get_actor_label()
                    for v in hit.to_tuple()
                    if isinstance(v, u.Actor)
                ]
            )
            assert "Building_071" in actor_hits, (last_d, actor_hits)
            records.append(
                dict(
                    image=name + ".png",
                    horizontal_m=last_d,
                    elapsed_s=elapsed,
                    center_ray_actor="Building_071",
                )
            )
            threshold = next(samples, None)
            if threshold is None:
                (out / "trajectory.json").write_text(json.dumps(records, indent=2))
                actors.destroy_actor(camera)
                u.unregister_slate_post_tick_callback(handle)
                u.SystemLibrary.quit_editor()
                return
        # Bound the per-frame motion even if exporting a frame or streaming assets stalls for a moment.
        distance = max(14, last_d - min(dt, 0.15) * 2.4)
        move(distance)
        last_d = distance
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc())
        u.unregister_slate_post_tick_callback(handle)
        u.SystemLibrary.quit_editor()


handle = u.register_slate_post_tick_callback(tick)
