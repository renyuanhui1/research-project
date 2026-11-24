"""Finalize ground contact, glazings, floating piers and terrain materials before verification."""

from pathlib import Path
import json
import sys
import math
import random
import unreal as u

work = Path(__file__).parent
sys.path.insert(0, str(work))
from harbor_level import (
    ROOT,
    MAP,
    ACTORS,
    LEVEL,
    ED,
    V,
    R,
    loc,
    obj,
    place,
    slab,
    surface,
    beach_z,
)
from harbor_import import material

tools = u.AssetToolsHelpers.get_asset_tools()
lib = u.MaterialEditingLibrary
lookup = {}
catalog = json.loads((work / "catalog.json").read_text(encoding="utf8"))
hulls = json.loads((work / "contact_points.json").read_text())
rng = random.Random(162)


def fixed_glass():
    folder = ROOT + "/Library/Materials"
    name = "MarineGlazing"
    if ED.does_asset_exist(folder + "/" + name):
        mat = obj(folder + "/" + name)
    else:
        mat = tools.create_asset(name, folder, u.Material, u.MaterialFactoryNew())
        mat.set_editor_property("blend_mode", u.BlendMode.BLEND_TRANSLUCENT)
        mat.set_editor_property(
            "translucency_lighting_mode",
            u.TranslucencyLightingMode.TLM_SURFACE_PER_PIXEL_LIGHTING,
        )
        mat.set_editor_property("two_sided", True)
        tint = lib.create_material_expression(mat, u.MaterialExpressionVectorParameter)
        tint.set_editor_property("parameter_name", "Tint")
        tint.set_editor_property("default_value", u.LinearColor(0.16, 0.22, 0.23, 1))
        assert lib.connect_material_property(tint, "", u.MaterialProperty.MP_BASE_COLOR)
        for value, prop in (
            (0.28, u.MaterialProperty.MP_OPACITY),
            (0.16, u.MaterialProperty.MP_ROUGHNESS),
        ):
            n = lib.create_material_expression(mat, u.MaterialExpressionConstant)
            n.set_editor_property("r", value)
            assert lib.connect_material_property(n, "", prop)
        lib.recompile_material(mat)
    changed = [mat]
    for f in (work / "library").glob("*/model.gltf"):
        spec = json.loads(f.read_text(encoding="utf8"))
        for desc in spec["materials"]:
            if (
                desc.get("extensions", {})
                .get("KHR_materials_transmission", {})
                .get("transmissionFactor", 0)
                < 0.45
                and desc.get("alphaMode") != "BLEND"
            ):
                continue
            mi = obj(
                ROOT
                + "/Library/"
                + f.parent.name.replace("-", "_")
                + "/MI_"
                + desc["name"]
            )
            lib.set_material_instance_parent(mi, mat)
            color = desc["pbrMetallicRoughness"].get(
                "baseColorFactor", [0.22, 0.29, 0.31, 1]
            )
            lib.set_material_instance_vector_parameter_value(
                mi, "Tint", u.LinearColor(*color)
            )
            changed.append(mi)
    assert u.EditorLoadingAndSavingUtils.save_packages(
        list({a.get_outermost() for a in changed}), True
    )


def soil():
    path = ROOT + "/Materials/M_SaltMeadow"
    if not ED.does_asset_exist(path):
        m = tools.create_asset(
            "M_SaltMeadow", ROOT + "/Materials", u.Material, u.MaterialFactoryNew()
        )

        def node(c):
            return lib.create_material_expression(m, c)

        def wire(a, out, b, i):
            assert lib.connect_material_expressions(a, out, b, i)

        def prop(a, out, p):
            assert lib.connect_material_property(a, out, p)

        xy = node(u.MaterialExpressionComponentMask)
        for c, v in (("r", True), ("g", True), ("b", False), ("a", False)):
            xy.set_editor_property(c, v)
        wire(node(u.MaterialExpressionWorldPosition), "", xy, "")
        folder = "/Game/MilitaryAirport/Landscape/LandscapeTextures/T_"

        def sample(path, period):
            uv = node(u.MaterialExpressionMultiply)
            uv.set_editor_property("const_b", 1 / period)
            wire(xy, "", uv, "A")
            n = node(u.MaterialExpressionTextureSample)
            n.set_editor_property("texture", obj(folder + path))
            if path.endswith("Normal"):
                n.set_editor_property(
                    "sampler_type", u.MaterialSamplerType.SAMPLERTYPE_NORMAL
                )
            wire(uv, "", n, "")
            return n

        grass = sample("Grass_b_BaseColor_Height", 560)
        sand = sample("Ground_a_BaseColor_Height", 420)
        noise = sample("VegetationVariation", 27000)
        blend = node(u.MaterialExpressionLinearInterpolate)
        wire(grass, "", blend, "A")
        wire(sand, "", blend, "B")
        wire(noise, "R", blend, "Alpha")
        tint = node(u.MaterialExpressionConstant3Vector)
        tint.set_editor_property("constant", u.LinearColor(0.68, 0.72, 0.58, 1))
        c = node(u.MaterialExpressionMultiply)
        wire(blend, "", c, "A")
        wire(tint, "", c, "B")
        prop(c, "", u.MaterialProperty.MP_BASE_COLOR)
        prop(sample("Grass_b_Normal", 560), "", u.MaterialProperty.MP_NORMAL)
        lib.recompile_material(m)
        ED.save_loaded_asset(m)
    surface(lookup["Backshore"], "SaltMeadow")
    distant = material(
        "Hinterland",
        "/Game/MilitaryAirport/Landscape/LandscapeTextures/T_Terrain_Far_BaseColor",
        tint=(0.61, 0.68, 0.66),
        world_scale=120000,
    )
    lookup["FarCoastline"].static_mesh_component.set_material(0, distant)
    # Raise the visual-only distant ridge slightly to submerge its seams with the survey terrain.
    lookup["DistantWater"].set_actor_scale3d(V(65000, 65000, 0.3))
    fog = lookup["SeaHaze"].component
    fog.set_editor_property("fog_density", 0.019)
    fog.set_editor_property("fog_height_falloff", 0.17)
    fog.set_editor_property(
        "fog_inscattering_luminance", u.LinearColor(0.43, 0.51, 0.56, 1)
    )


def set_contact():
    audit = {}
    for row in catalog:
        if row["state"] not in ("aground", "overturned"):
            continue
        a = lookup[row["id"]]
        tf = a.get_actor_transform()
        key = row["model"].replace("-", "_")
        pts = [tf.transform_location(V(*v)) for v in hulls[key]]

        def diff(v):
            return v.z / 100 - (
                beach_z(v.x / 100, v.y / 100) if row["state"] == "aground" else 0.09
            )

        delta = min(diff(v) for v in pts)
        target = -0.15 if row["state"] == "aground" else -0.025
        shift = target - delta
        a.set_actor_location(
            a.get_actor_location() + V(0, 0, shift * 100), False, False
        )
        audit[row["id"]] = {
            "initial_min_clearance_m": round(delta, 3),
            "vertical_adjustment_m": round(shift, 3),
            "final_min_clearance_m": round(delta + shift, 3),
            "support_vertices": len(pts),
        }
    (work / "ground-contact.json").write_text(json.dumps(audit, indent=2))


def piers():
    for j in range(5):
        x = -800 + j * 162
        pier = lookup["Pier_%02d" % j]
        pier.set_actor_scale3d(V(9, 160, 0.64))
        pier.set_actor_location(loc(x, -329, -2.04), False, False)
        a = slab(
            "Gangway_%d" % j, x, -252, -0.52, 3.1, 16, 0.18, "Pier", "DockEquipment"
        )
        a.set_actor_rotation(R(roll=6.7), False)
        for y in range(-268, -409, -20):
            for sy in (-1, 1):
                a = place(
                    "/Engine/BasicShapes/Cylinder",
                    (x + 4.25 * sy, y, -2.21),
                    "TimberPiles",
                    scale=(0.46, 0.46, 3.9),
                )
                surface(a, "Pier")
    for a in lookup.values():
        if str(a.get_folder_path()).endswith("Moorings"):
            p = a.get_actor_location()
            p.z -= 189
            a.set_actor_location(p, False, False)
    for i in range(1, 26):
        a = lookup["Vessel_%03d" % i]
        center, e = a.get_actor_bounds(False)
        x = center.x / 100
        y = center.y / 100
        pier = -800 + ((i - 1) // 5) * 162
        side = 1 if x > pier else -1
        for end in (-1, 1):
            v1 = V(
                (x - side * e.x / 100 * 0.75) * 100, y * 100 + end * 0.58 * e.y, -156
            )
            v2 = V((pier + side * 3.6) * 100, y * 100 + end * (e.y * 0.71 + 125), -150)
            mid = (v1 + v2) * 0.5
            mid.z -= 23
            for p, q in ((v1, mid), (mid, v2)):
                delta = q - p
                length = delta.length()
                rope = place(
                    "/Engine/BasicShapes/Cylinder",
                    tuple(
                        v / 100
                        for v in ((p.x + q.x) / 2, (p.y + q.y) / 2, (p.z + q.z) / 2)
                    ),
                    "MooringLines",
                    scale=(0.045, 0.045, length / 100),
                )
                rope.set_actor_rotation(u.MathLibrary.make_rot_from_z(delta), False)
                rope.set_actor_enable_collision(False)
                surface(rope, "Pier")


def roof():
    folder = ROOT + "/Meshes"
    path = work / "terrain/ShedTornRoof.fbx"
    if not ED.does_asset_exist(folder + "/" + path.stem):
        t = u.AssetImportTask()
        op = u.FbxImportUI()
        op.set_editor_property("automated_import_should_detect_type", False)
        op.set_editor_property("mesh_type_to_import", u.FBXImportType.FBXIT_STATIC_MESH)
        op.set_editor_property("import_materials", False)
        op.set_editor_property("import_textures", False)
        op.static_mesh_import_data.set_editor_property(
            "vertex_color_import_option", u.VertexColorImportOption.REPLACE
        )
        for key, val in (
            ("filename", str(path)),
            ("destination_path", folder),
            ("options", op),
            ("automated", True),
            ("save", False),
        ):
            t.set_editor_property(key, val)
        tools.import_asset_tasks([t])
    m = obj(folder + "/" + path.stem)
    slots = list(m.static_materials)
    for slot in slots:
        slot.set_editor_property(
            "material_interface",
            obj(ROOT + "/Materials/M_" + str(slot.material_slot_name)),
        )
    m.set_editor_property("static_materials", slots)
    m.get_editor_property("body_setup").set_editor_property(
        "collision_trace_flag", u.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
    )
    for i in (95, 98):
        lookup["Building_%03d" % i].static_mesh_component.set_static_mesh(m)
    packs = [m.get_outermost()]
    for k in range(4):
        rock = obj(ROOT + "/Backdrop/Armour" + str(k))
        rock.get_editor_property("body_setup").set_editor_property(
            "collision_trace_flag", u.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
        )
        rock.modify()
        packs.append(rock.get_outermost())
    assert u.EditorLoadingAndSavingUtils.save_packages(packs, False)


def main():
    assert LEVEL.load_level(MAP)
    lookup.update({a.get_actor_label(): a for a in ACTORS.get_all_level_actors()})
    if "Gangway_0" in lookup:
        raise RuntimeError("Detail pass already applied")
    fixed_glass()
    soil()
    set_contact()
    piers()
    roof()
    for row in catalog:
        p = lookup[row["id"]].get_actor_location()
        row["xyz_m"] = [round(v / 100, 3) for v in (p.x, p.y, p.z)]
    assert LEVEL.save_current_level()
    (work / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf8")
    u.log("HARBOR_DETAILS_SAVED")


main()
