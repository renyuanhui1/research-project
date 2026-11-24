"""Open saved coastal map in a rendering UE editor. Capture views and check the on-disk level."""

import json
import math
import time
import traceback
from collections import Counter
from pathlib import Path

import unreal as u

ROOT = Path(__file__).parent
MAP = "/Game/CoastalHarbor/Maps/CoastalHarbor"
actors = u.get_editor_subsystem(u.EditorActorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
level = u.get_editor_subsystem(u.LevelEditorSubsystem)
render = u.RenderingLibrary
out = ROOT / "review"
out.mkdir(exist_ok=True)
catalog = {
    v["id"]: v for v in json.loads((ROOT / "catalog.json").read_text(encoding="utf8"))
}


def view(name, eye, aim):
    v = u.Vector(*(float(n) * 100 for n in eye))
    target = u.Vector(*(float(n) * 100 for n in aim))
    return name, v, u.MathLibrary.find_look_at_rotation(v, target)


def target_view(name, key, offset, relative_aim=(0, 0, 0)):
    a = lookup[key]
    center, extent = a.get_actor_bounds(False)
    c = [v / 100 for v in (center.x, center.y, center.z)]
    eye = [v + d for v, d in zip(c, offset)]
    aim = [v + d for v, d in zip(c, relative_aim)]
    return view(name, eye, aim)


def hull_view(name, key):
    a = lookup[key]
    c, e = a.get_actor_bounds(False)
    length = max(e.x, e.y) * 0.02
    forward = a.get_actor_forward_vector()
    side = a.get_actor_right_vector()
    focus = c - forward * (length * 10) - u.Vector(0, 0, e.z * 0.24)
    eye = (
        focus
        - side * (length * 97)
        - forward * (length * 40)
        + u.Vector(0, 0, length * 27)
    )
    return name, eye, u.MathLibrary.find_look_at_rotation(eye, focus)


def audit():
    if editor.get_editor_world().get_path_name().split(".")[0] != MAP:
        if not level.load_level(MAP):
            raise RuntimeError("failed to open saved map")
    all_actors = actors.get_all_level_actors()
    by_label = {a.get_actor_label(): a for a in all_actors}
    count = Counter()
    inspected = 0
    sizes = {}
    for key, row in catalog.items():
        a = by_label[key]
        tags = {str(t) for t in a.tags}
        assert key in tags and row["category"] in tags
        inspected += int("Inspection" in tags)
        center, extent = a.get_actor_bounds(False)
        mx = max(extent.x, extent.y) * 0.02
        assert 1 < mx < 80, (key, mx)
        assert max(abs(center.x), abs(center.y)) < 100000, key
        row["ue_actor_name"] = a.get_name()
        row["ue_actor_path"] = a.get_path_name()
        row["damage_status"] = (
            "intact" if row["visible_evidence"].startswith("intact") else "damaged"
        )
        count[row["category"]] += 1
        sizes[key] = round(mx, 2)
    assert inspected == 32 and count == {"Building": 120, "Vessel": 40, "Vehicle": 120}
    redesigned = "PortSurveyExposure" in by_label
    surfaces = (
        ("Backshore", "Foreshore")
        if redesigned
        else ("Backshore", "Foreshore", "HarborWater")
    )
    for label in surfaces:
        a = by_label[label]
        c, e = a.get_actor_bounds(False)
        assert abs(e.x - 100000) < 3, (label, e)
    world = editor.get_editor_world()
    mode = (
        world.get_world_settings().get_editor_property("default_game_mode").get_name()
    )
    assert "ProjectAirSim" in mode, mode
    data = dict(
        targets=dict(count),
        inspection_targets=inspected,
        afloat=sum(r["state"] == "afloat" for r in catalog.values()),
        aground=sum(r["state"] == "aground" for r in catalog.values()),
        actor_count=len(all_actors),
        game_mode=mode,
        width_m=2000,
        depth_m=2000,
        model_span_m=sizes,
    )
    probes = {}
    for tag, x, y, top in (
        ("land", 100 if redesigned else 8, -95 if redesigned else 5, 0),
        ("open_water", 105, -625, -2.65),
        ("quay", -505, -155, 0.17),
        ("pier", -800, -335, -1.72),
        (
            "street",
            -700 if redesigned else 12,
            -118.4 if redesigned else 290,
            0.105 if redesigned else 0.09,
        ),
    ):
        hit = u.SystemLibrary.line_trace_single(
            world,
            u.Vector(x * 100, y * 100, 3000),
            u.Vector(x * 100, y * 100, -1500),
            u.TraceTypeQuery.TRACE_TYPE_QUERY1,
            True,
            [],
            u.DrawDebugTrace.NONE,
            True,
        )
        if not hit:
            raise RuntimeError("No visibility collision on " + tag)
        parts = hit.to_tuple()
        # Blueprint BreakHitResult returns (blocking, penetration, time, distance, location, impact point, ...).
        observed = parts[5].z / 100
        probes[tag] = {"expected_m": top, "measured_m": round(observed, 3)}
        assert abs(observed - top) < 0.12, (tag, observed, top)
    if (ROOT / "street-cameras.json").exists() or redesigned:
        frontage = [
            a
            for a in all_actors
            if str(a.get_folder_path()) == "Harbor/StreetFrontages"
        ]
        assert all("Inspection" not in {str(t) for t in a.tags} for a in frontage)
        inventory = Counter()
        for a in frontage:
            mesh = a.static_mesh_component.static_mesh
            if any(folder in mesh.get_path_name() for folder in ("/StreetArchitecture/", "/StormAftermath/")):
                inventory[mesh.get_name()] += 1
                assert all(s.material_interface for s in mesh.static_materials)
        data["street_architecture"] = dict(inventory)
        data["street_folder_actors"] = len(frontage)
        roofs = {}
        for name in ("Building_071", "Building_085", "Building_098"):
            a = by_label[name]
            c, e = a.get_actor_bounds(False)
            hit = u.SystemLibrary.line_trace_single(
                world,
                c + u.Vector(0, 0, 3000),
                c - u.Vector(0, 0, 1500),
                u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                True,
                [],
                u.DrawDebugTrace.NONE,
                True,
            )
            assert hit and a in hit.to_tuple(), (name, hit)
            roofs[name] = round(hit.to_tuple()[5].z / 100, 3)
        data["roof_collision_z_m"] = roofs
    if redesigned:
        vol = by_label["PortSurveyExposure"]
        pp = vol.get_editor_property("settings")
        data["level_exposure"] = {
            k: pp.get_editor_property("auto_exposure_" + k)
            for k in ("min_brightness", "max_brightness", "bias")
        }
        assert vol.get_editor_property("unbound") and all(
            pp.get_editor_property("override_auto_exposure_" + k)
            for k in data["level_exposure"]
        )
        assert (
            data["level_exposure"]["min_brightness"]
            == data["level_exposure"]["max_brightness"]
            == 1.25
        )
        if pp.get_editor_property("override_dynamic_global_illumination_method"):
            assert (
                pp.get_editor_property("dynamic_global_illumination_method")
                == u.DynamicGlobalIlluminationMethod.NONE
            )
            data["level_diffuse_gi"] = "None"
            sky = by_label["MarineSkyLight"].light_component
            data["skylight_intensity"] = sky.get_editor_property("intensity")
        data["fog_density"] = by_label["SeaHaze"].component.get_editor_property(
            "fog_density"
        )
        plan = json.loads((ROOT / "plan.json").read_text())
        extent = plan.get("surface_half_extent_m")
        traces = {}
        for name, x, y in [
            ("west_in", -997, 80),
            ("west_out", -1003, 80),
            ("east_in", 997, 60),
            ("east_out", 1003, 60),
            ("north_in", 719, 997),
            ("north_out", 719, 1003),
            ("outer_water", 900 if plan.get("outline_m") else extent - 10 if extent else 2000, -900),
        ]:
            hit = u.SystemLibrary.line_trace_single(
                world,
                u.Vector(x * 100, y * 100, 14000),
                u.Vector(x * 100, y * 100, -1500),
                u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                True,
                [],
                u.DrawDebugTrace.NONE,
                True,
            )
            assert hit, name
            traces[name] = round(hit.to_tuple()[5].z / 100, 3)
        for side in ("west", "east", "north"):
            assert abs(traces[side + "_in"] - traces[side + "_out"]) < 1.0, (
                side,
                traces,
            )
        data["terrain_boundary_z_m"] = traces
        water = by_label["HarborWater"]
        mesh_box = water.static_mesh_component.static_mesh.get_bounding_box()
        data["water_span_m"] = round((mesh_box.max.x - mesh_box.min.x) * water.get_actor_scale3d().x / 100)
        if extent:
            bounds = plan.get("water_bounds_m", [-extent, -extent, extent, extent])
            assert abs(data["water_span_m"] - (bounds[2] - bounds[0])) <= 0.51
            for x, y in [(extent + 30, 500), (-extent - 30, 500),
                         (0, extent + 30), (0, -extent - 30)]:
                hit = u.SystemLibrary.line_trace_single(
                    world, u.Vector(x * 100, y * 100, 15000),
                    u.Vector(x * 100, y * 100, -2000), u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                    True, [], u.DrawDebugTrace.NONE, True,
                )
                assert not hit, (x, y, hit)
            data["exterior_probes_clear"] = 4
        bykey = {a.get_name(): a for a in all_actors}
        # Editor-only visualizers may enlarge Blueprint bounds. The native build also
        # checks geometric centers, and the rendered reload verifies every saved pivot.
        reference = json.loads((ROOT / "parcel-check.json").read_text())
        data["max_planned_center_error_m"] = reference["max_center_error_m"]
        pivot_errors = []
        for key, xyz in reference["pivots"].items():
            pos = bykey[key].get_actor_location()
            pivot_errors.append(math.dist([pos.x / 100, pos.y / 100, pos.z / 100], xyz))
        data["max_reloaded_pivot_error_m"] = max(pivot_errors)
        assert data["max_reloaded_pivot_error_m"] < 0.02
        trees = Counter()
        for a in all_actors:
            if isinstance(a, u.InstancedFoliageActor):
                for comp in a.get_components_by_class(
                    u.HierarchicalInstancedStaticMeshComponent
                ):
                    mesh = comp.static_mesh
                    if mesh:
                        trees[mesh.get_name()] += comp.get_instance_count()
        data["instanced_trees"] = dict(trees)
        assert sum(trees.values()) == sum(
            item[0] == "tree" for item in plan["decor"]
        ), trees
        road_actor = by_label["TownStreets"]
        ignored = [by_label[k] for k in catalog]
        connected = {}
        for road in plan["roads"]:
            pts = road["xy"]
            x, y = pts[len(pts) // 2]
            hit = u.SystemLibrary.line_trace_single(
                world,
                u.Vector(x * 100, y * 100, 14000),
                u.Vector(x * 100, y * 100, -1500),
                u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                True,
                ignored,
                u.DrawDebugTrace.NONE,
                True,
            )
            assert hit and road_actor in hit.to_tuple(), (road["name"], str(hit))
            connected[road["name"]] = round(hit.to_tuple()[5].z / 100, 3)
        clearance = []
        for dx, dy in ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)):
            h = u.SystemLibrary.line_trace_single(
                world,
                u.Vector((-390 + dx) * 100, (-154 + dy) * 100, 7500),
                u.Vector((-390 + dx) * 100, (-154 + dy) * 100, -150),
                u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                True,
                [],
                u.DrawDebugTrace.NONE,
                True,
            )
            assert h and by_label["Quayside"] in h.to_tuple(), (dx, dy, h)
            clearance.append(round(h.to_tuple()[5].z / 100, 3))
        data["spawn_ground_z_m"] = clearance
        data["road_surface_probes_m"] = connected
    for row in catalog.values():
        actor = by_label[row["id"]]
        pos = actor.get_actor_location()
        row["xyz_m"] = [round(a / 100, 3) for a in (pos.x, pos.y, pos.z)]
    data["vertical_collision_probes"] = probes
    data["vehicles_by_model"] = dict(
        Counter(v["kind"] for v in catalog.values() if v["category"] == "Vehicle")
    )
    (out / "targets.json").write_text(
        json.dumps(list(catalog.values()), indent=2), encoding="utf8"
    )
    (out / "check.json").write_text(json.dumps(data, indent=2), encoding="utf8")
    return by_label


def setup():
    global lookup, world, camera, cap, rt, views
    u.AutomationLibrary.finish_loading_before_screenshot()
    lookup = audit()
    world = editor.get_editor_world()
    if (ROOT / "street-cameras.json").exists() or "PortSurveyExposure" in lookup:
        u.SystemLibrary.execute_console_command(
            world, "r.Shadow.Virtual.MaxPhysicalPagesSceneCapture 1536"
        )
    camera = actors.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0, 0, 3000))
    cap = camera.get_component_by_class(u.SceneCaptureComponent2D)
    rt = render.create_render_target2d(
        world, 1440, 900, u.TextureRenderTargetFormat.RTF_RGBA8
    )
    if "PortSurveyExposure" in lookup:
        rt.set_editor_property("target_gamma", 2.2)
    cap.set_editor_property("texture_target", rt)
    cap.set_editor_property("capture_source", u.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    cap.set_editor_property("fov_angle", 72)
    cap.set_editor_property("capture_every_frame", True)
    cap.set_editor_property("always_persist_rendering_state", True)
    views = [
        view("01_harbor", (-1000, -758, 226), (-395, -128, 0)),
        view("02_town", (-518, -280, 102), (-265, 175, 9)),
        target_view("03_roof", "Building_098", (22, 17, 22)),
        hull_view("04_aground", "Vessel_035"),
        target_view("05_vehicle", "Vehicle_002", (4.5, 4, 2.4)),
        hull_view("06_afloat", "Vessel_002"),
        target_view("07_roadside", "Building_071", (7, 20, 6)),
        target_view("08_truck", "Vehicle_092", (10, 8, 5)),
        view("09_neighborhood", (-830, 204, 26), (-748, 267, 4)),
        view("10_quayside", (-835, -412, 24), (-773, -280, 3)),
    ]
    overrides = ROOT / "street-cameras.json"
    if overrides.exists():
        custom = {item[0]: view(*item) for item in json.loads(overrides.read_text())}
        views = [custom.pop(v[0], v) for v in views]
        views.extend(custom.values())
    if "PortSurveyExposure" in lookup:
        station = json.loads((ROOT / "market-stations.json").read_text())[
            "Building_071"
        ]
        cap.set_editor_property("fov_angle", 68)
        views = [
            view("13_survey", (-1380, -380, 1170), (0, 0, 0)),
            view("14_harbour_quarters", (-1080, -355, 232), (-525, 70, 4)),
            view("07_roadside", station["eye"], station["aim"]),
            target_view("03_roof", "Building_098", (22, 17, 22)),
            view("15_old_town", (-775, 31, 51), (-544, 160, 4)),
            view("16_valley", (465, 730, 158), (-235, 360, 20)),
            hull_view("04_aground", "Vessel_035"),
            target_view("08_truck", "Vehicle_092", (10, 8, 5)),
        ]
    next_view()


def next_view():
    global since
    name, pos, rot = views[0]
    camera.set_actor_location_and_rotation(pos, rot, False, False)
    u.EditorLevelLibrary.set_level_viewport_camera_info(pos, rot)
    since = time.monotonic()


def stop():
    u.unregister_slate_post_tick_callback(handle)
    u.SystemLibrary.quit_editor()


def tick(_dt):
    global stage, since
    try:
        if time.monotonic() - since < 4:
            return
        if stage == 0:
            setup()
            stage = 1
            return
        if stage == 1:
            # Complete texture residency and exposure settling before reading GPU output.
            if time.monotonic() - since < 10:
                return
            name = views[0][0]
            render.export_render_target(world, rt, str(out), name + ".png")
            if not (out / (name + ".png")).exists():
                raise RuntimeError(name + " missing")
            u.log("HARBOR_CAPTURE " + name)
            views.pop(0)
            if views:
                next_view()
            else:
                actors.destroy_actor(camera)
                (out / "complete.txt").write_text(
                    "Reload, inventory and GPU renders completed."
                )
                stop()
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf8")
        stop()


stage = 0
since = time.monotonic()
handle = u.register_slate_post_tick_callback(tick)
