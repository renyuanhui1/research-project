"""Apply the planned port town to the previous CoastalHarbor map in UE 5.2."""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import unreal as u

work = Path(__file__).parent
sys.path.insert(0, str(work))
from harbor_import import material, land
from harbor_level import (
    ROOT,
    MAP,
    LEVEL,
    ACTORS,
    ED,
    PROPS,
    V,
    R,
    obj,
    place,
    loc,
    fitted,
)

dest = ROOT + "/PortTown"
kit = ROOT + "/StreetArchitecture/"
src = (
    Path(u.Paths.convert_relative_path_to_full(u.Paths.project_dir()))
    / "SourceArt/CoastalHarbor/PortTown"
)
tools = u.AssetToolsHelpers.get_asset_tools()


def coast_material(name, low, high, shore=False):
    """World-space microtexture fades to broad soil variation at survey altitudes."""
    asset = ROOT + "/Materials/" + name
    if ED.does_asset_exist(asset):
        return obj(asset)
    m = tools.create_asset(
        name, ROOT + "/Materials", u.Material, u.MaterialFactoryNew()
    )
    lib = u.MaterialEditingLibrary

    def node(k):
        return lib.create_material_expression(m, k)

    def wire(a, out, b, slot):
        assert lib.connect_material_expressions(a, out, b, slot)

    def color(rgb):
        v = node(u.MaterialExpressionConstant3Vector)
        v.set_editor_property("constant", u.LinearColor(*rgb, 1))
        return v

    def scaled(source, ch, value):
        mul = node(u.MaterialExpressionMultiply)
        wire(source, ch, mul, "A")
        mul.set_editor_property("const_b", value)
        return mul

    def mix(a, b, weight, output=""):
        q = node(u.MaterialExpressionLinearInterpolate)
        wire(a, "", q, "A")
        wire(b, "", q, "B")
        if isinstance(weight, (int, float)):
            q.set_editor_property("const_alpha", weight)
        else:
            wire(weight, output, q, "Alpha")
        return q

    wp = node(u.MaterialExpressionWorldPosition)
    xy = node(u.MaterialExpressionComponentMask)
    for k in "rgba":
        xy.set_editor_property(k, k in "rg")
    wire(wp, "", xy, "")

    def sample(path, period, normal=False):
        t = node(u.MaterialExpressionTextureSample)
        t.set_editor_property("texture", obj(path))
        if normal:
            t.set_editor_property(
                "sampler_type", u.MaterialSamplerType.SAMPLERTYPE_NORMAL
            )
        wire(scaled(xy, "", 1 / period), "", t, "")
        return t

    mask = sample(ROOT + "/Backdrop/MeadowMask", 71400)
    far = mix(color(low), color(high), mask, "R")
    scan = kit + "Ground/" + ("YardStone" if shore else "MaritimeSward")
    close = mix(far, sample(scan + "_BC", 680 if shore else 1750), 0.27)
    fall = node(u.MaterialExpressionSaturate)
    wire(scaled(node(u.MaterialExpressionPixelDepth), "", 1 / 37500), "", fall, "")
    base = mix(close, far, fall)
    if shore:
        z = node(u.MaterialExpressionComponentMask)
        for k in "rgba":
            z.set_editor_property(k, k == "b")
        wire(wp, "", z, "")
        weight = node(u.MaterialExpressionAdd)
        wire(scaled(z, "", 1 / 175), "", weight, "A")
        weight.set_editor_property("const_b", 1)
        clamp = node(u.MaterialExpressionSaturate)
        wire(weight, "", clamp, "")
        bankcolor = mix(
            color((0.076, 0.092, 0.050)), color((0.14, 0.155, 0.085)), mask, "R"
        )
        base = mix(base, bankcolor, clamp)
    assert lib.connect_material_property(base, "", u.MaterialProperty.MP_BASE_COLOR)
    rough = node(u.MaterialExpressionConstant)
    rough.set_editor_property("r", 0.86)
    assert lib.connect_material_property(rough, "", u.MaterialProperty.MP_ROUGHNESS)
    normal = sample(scan + "_N", 680 if shore else 1750, True)
    softened = mix(normal, color((0, 0, 1)), fall)
    assert lib.connect_material_property(softened, "", u.MaterialProperty.MP_NORMAL)
    m.set_editor_property("two_sided", True)
    lib.recompile_material(m)
    ED.save_loaded_asset(m, False)
    return m


def imports(files=None):
    skins = {
        "CoastalCutRock": material("CoastalCutRock", land + "Cliff_BaseColor_Height",
                                   land + "Cliff_Normal", rough=0.88),
        "MixedCoastalGrass": coast_material(
            "M_PortGrass", (0.076, 0.092, 0.050), (0.14, 0.155, 0.085)
        ),
        "MeadowPasture": coast_material(
            "M_PortPasture", (0.069, 0.088, 0.048), (0.122, 0.145, 0.080)
        ),
        "MeadowStubble": coast_material(
            "M_PortStubble", (0.118, 0.103, 0.071), (0.161, 0.149, 0.098)
        ),
        "MeadowHay": coast_material(
            "M_PortHay", (0.094, 0.107, 0.057), (0.143, 0.153, 0.081)
        ),
        "Mud": coast_material(
            "M_PortTidalSilt", (0.064, 0.061, 0.050), (0.111, 0.113, 0.092), True
        ),
    }
    for f in sorted(src.glob("*.fbx") if files is None else files):
        path = dest + "/" + f.stem
        if not ED.does_asset_exist(path) or (work / "refresh-geometry").exists():
            t = u.AssetImportTask()
            t.replace_existing = True
            t.replace_existing_settings = True
            opt = u.FbxImportUI()
            opt.automated_import_should_detect_type = False
            opt.mesh_type_to_import = u.FBXImportType.FBXIT_STATIC_MESH
            opt.import_materials = opt.import_textures = opt.import_as_skeletal = False
            data = opt.static_mesh_import_data
            data.combine_meshes = True
            data.auto_generate_collision = False
            data.generate_lightmap_u_vs = False
            t.filename, t.destination_path, t.automated, t.save, t.options = (
                str(f),
                dest,
                True,
                True,
                opt,
            )
            tools.import_asset_tasks([t])
        mesh = obj(path)
        for i, sm in enumerate(mesh.static_materials):
            slot = str(sm.material_slot_name)
            mesh.set_material(
                i, skins[slot] if slot in skins else obj(ROOT + "/Materials/M_" + slot)
            )
        mesh.get_editor_property("body_setup").set_editor_property(
            "collision_trace_flag", u.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
        )
        ED.save_loaded_asset(mesh, False)


def apply_waterfront(water):
    water.modify()
    component = water.static_mesh_component
    component.modify()
    water.set_actor_location(V(), False, False)
    water.set_actor_scale3d(V(1, 1, 1))
    component.set_static_mesh(obj(dest + "/CoastalWater"))
    component.set_editor_property("bounds_scale", 2.0)
    place(obj(dest + "/TerrainCutFaces"), (0, 0, 0), "PortQuarter", "TerrainCutFaces")


def shift(a, row, site):
    r = a.get_actor_rotation()
    r.yaw = row["yaw"] + site["delta"]
    a.set_actor_rotation(r, False)
    c, e = a.get_actor_bounds(False)
    p = a.get_actor_location()
    x, y = site["xy"]
    a.set_actor_location(
        p + loc(x, y, site["base"]) - V(c.x, c.y, c.z - e.z), False, False
    )


def light(bylabel):
    fog = bylabel["SeaHaze"].component
    fog.set_editor_property("fog_density", 0.0065)
    bylabel["CoastalSun"].light_component.set_editor_property("light_source_angle", 1.1)
    bylabel["MarineSkyLight"].light_component.set_editor_property("intensity", 0.80)
    vol = bylabel.get("PortSurveyExposure") or place(
        u.PostProcessVolume, (0, 0, 180), "Environment", "PortSurveyExposure"
    )
    vol.set_editor_property("unbound", True)
    vol.set_editor_property("priority", 20.0)
    pp = vol.get_editor_property("settings")
    pp.set_editor_property("override_dynamic_global_illumination_method", True)
    pp.set_editor_property(
        "dynamic_global_illumination_method", u.DynamicGlobalIlluminationMethod.NONE
    )
    for name, val in [
        ("min_brightness", 1.25),
        ("max_brightness", 1.25),
        ("bias", 1.0),
    ]:
        pp.set_editor_property("override_auto_exposure_" + name, True)
        pp.set_editor_property("auto_exposure_" + name, val)
    vol.set_editor_property("settings", pp)


def plants(world, rows, species, half_extent):
    rng = random.Random(515)
    if species is not None and len(species) != len(rows):
        raise ValueError("Tree species and planting positions must correspond")
    groups = defaultdict(list)
    for i, (_, x, y, angle, span, z) in enumerate(rows):
        key = species[i] if species is not None else (
            "lombardy_poplar" if rng.random() < 0.17 else "silver_birch"
        )
        if max(abs(x), abs(y)) <= half_extent:
            groups[key].append((x, y, z, angle, span))
    for key in ("lombardy_poplar", "silver_birch"):
        far = dest + "/FT_" + key + "_Distant"
        if ED.does_asset_exist(far):
            u.InstancedFoliageActor.remove_all_instances(world, obj(far))
        trees = groups[key]
        mesh = obj(ROOT + "/Library/" + key + "/" + key)
        bounds = mesh.get_bounding_box()
        width = max(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y)
        tn = "FT_" + key
        path = dest + "/" + tn
        ft = (
            obj(path)
            if ED.does_asset_exist(path)
            else tools.create_asset(
                tn,
                dest,
                u.FoliageType_InstancedStaticMesh,
                u.FoliageType_InstancedStaticMeshFactory(),
            )
        )
        if ft.get_editor_property("mesh") != mesh:
            ft.set_editor_property("mesh", mesh)
        ED.save_loaded_asset(ft, True)
        ts = []
        for x, y, z, yaw, span in trees:
            scale = 100 * span / width
            tf = u.Transform()
            tf.translation = loc(x, y, z) - V(0, 0, bounds.min.z * scale)
            tf.rotation = R(yaw=yaw).quaternion()
            tf.scale3d = V(scale, scale, scale)
            ts.append(tf)
        u.InstancedFoliageActor.remove_all_instances(world, ft)
        if ts:
            u.InstancedFoliageActor.add_instances(world, ft, ts)


def apply_quay(bylabel, plan):
    quay = bylabel["Quayside"]
    quay.set_actor_scale3d(V(1, 1, 1))
    quay.set_actor_rotation(R(), False)
    quay.set_actor_location(V(), False, False)
    quay.static_mesh_component.set_static_mesh(obj(dest + "/WorkingQuay"))
    quay.static_mesh_component.set_editor_property("override_materials", [])
    for key, x, y, z, size, yaw in plan["quay_props"]:
        source = (
            ROOT + "/Library/" + key + "/" + key
            if key in ("warehouse", "boathouse")
            else PROPS + key
        )
        fitted(source, (x, y, z), "PortQuarter", size, yaw=yaw)


def verify_parcels(plan):
    bykey = {a.get_name(): a for a in ACTORS.get_all_level_actors()}
    pivots = {}
    worst = 0.0
    for key, site in plan["placements"].items():
        a = bykey[key]
        c, e = a.get_actor_bounds(False)
        worst = max(worst, math.dist([c.x / 100, c.y / 100], site["xy"]))
        pos = a.get_actor_location()
        pivots[key] = [pos.x / 100, pos.y / 100, pos.z / 100]
    assert worst < 0.02, worst
    (work / "parcel-check.json").write_text(
        json.dumps({"max_center_error_m": worst, "pivots": pivots}, indent=1)
    )


def main():
    assert LEVEL.load_level(MAP)
    world = u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
    acts = ACTORS.get_all_level_actors()
    bylabel = {a.get_actor_label(): a for a in acts}
    bykey = {a.get_name(): a for a in acts}
    for a in acts:
        if str(a.get_folder_path()) == "Harbor/PortQuarter":
            ACTORS.destroy_actor(a)
    plan = json.loads((work / "plan.json").read_text())
    original = {r["key"]: r for r in json.loads((work / "inventory.json").read_text())}
    assert len(plan["lots"]) == 362
    imports()
    for key, site in plan["placements"].items():
        shift(bykey[key], original[key], site)
    for r in original.values():
        if r["key"] in plan["placements"] or r["key"] not in bykey:
            continue
        if (
            not u.SystemLibrary.is_valid(bykey[r["key"]])
            or bykey[r["key"]].get_actor_label() != r["label"]
        ):
            continue
        if (
            r["folder"]
            in (
                "StreetFrontages",
                "Streets",
                "RoadMarkings",
                "ResidentialLots",
                "GardenTrees",
                "GardenEdges",
                "Vegetation",
                "Shelterbelt",
                "Lights",
                "StreetFurniture",
                "StormDebris",
            )
            or r["label"].startswith("ServiceYard_")
            or r["label"] in ("DistantWater", "FarCoastline")
        ):
            ACTORS.destroy_actor(bykey[r["key"]])
    for name, asset in [("Backshore", "TownGround"), ("Foreshore", "TidalShelf")]:
        a = bylabel[name]
        a.set_actor_rotation(R(), False)
        a.set_actor_scale3d(V(1, 1, 1))
        a.set_actor_location(V(), False, False)
        a.static_mesh_component.set_static_mesh(obj(dest + "/" + asset))
        a.static_mesh_component.set_editor_property("override_materials", [])
    water = bylabel["HarborWater"]
    half_extent = plan.get("surface_half_extent_m", 1120)
    apply_waterfront(water)
    for asset in ("CoastBeyondTown", "TownStreets", "TownLots", "FieldParcels"):
        place(obj(dest + "/" + asset), (0, 0, 0), "PortQuarter", asset)
    light(bylabel)
    trees = []
    for key, x, y, angle, span, z in plan["decor"]:
        if key == "tree":
            trees.append((key, x, y, angle, span, z))
            continue
        if key.startswith("SM_"):
            fitted(PROPS + key[3:], (x, y, z + 0.07), "PortQuarter", span, yaw=angle)
        else:
            place(obj(kit + key), (x, y, z + 0.09), "PortQuarter", yaw=angle)
    plants(world, trees, plan.get("tree_species"), half_extent)
    apply_quay(bylabel, plan)
    # Map each original inspection station into the moved target's local street frame.
    cameras = {}
    for label, eye, aim in [
        ("Building_071", (-825.5, 78.5, 14.2), (-830, 62, 5.60)),
        ("Building_085", (-538, 91, 11), (-540, 111, 5.6)),
    ]:
        a = bylabel[label]
        c, e = a.get_actor_bounds(False)
        old = original[a.get_name()]
        delta = math.radians(plan["placements"][a.get_name()]["delta"])
        pos = a.get_actor_location()

        def moved(q):
            dx, dy = q[0] - old["p"][0], q[1] - old["p"][1]
            return [
                round(pos.x / 100 + dx * math.cos(delta) - dy * math.sin(delta), 4),
                round(pos.y / 100 + dx * math.sin(delta) + dy * math.cos(delta), 4),
                round(pos.z / 100 + q[2] - old["p"][2], 4),
            ]

        normal_sign = 1 if eye[1] > aim[1] else -1
        cameras[label] = {
            "eye": moved(eye),
            "aim": moved(aim),
            "front_normal": [
                -normal_sign * math.sin(delta),
                normal_sign * math.cos(delta),
            ],
        }
    verify_parcels(plan)
    assert LEVEL.save_current_level()
    (work / "market-stations.json").write_text(json.dumps(cameras, indent=2))
    u.log("PORTTOWN_SAVED " + str(len(trees)) + " trees")


if __name__ == "__main__":
    main()
