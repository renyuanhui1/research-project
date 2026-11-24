"""Dress the coastal level using civilian PBR models and keep its reference catalog synchronized."""

import json
import math
import random
import sys
from pathlib import Path
import unreal as u

work = Path(__file__).parent
sys.path.insert(0, str(work))
import harbor_level as layout
from harbor_level import (
    MAP,
    ROOT,
    ED,
    LEVEL,
    ACTORS,
    V,
    R,
    loc,
    obj,
    place,
    slab,
    surface,
    local_mesh,
    shore,
    beach_z,
)
from harbor_import import material

rng = random.Random(7331)
lib = u.MaterialEditingLibrary
tools = u.AssetToolsHelpers.get_asset_tools()
catalog = json.loads((work / "catalog.json").read_text(encoding="utf8"))
lookup = {}


def library(key, storm=False):
    name = key.replace("-", "_")
    return ROOT + "/Library/" + name + "/" + name + ("_Storm" if storm else "")


def align_mesh(a, xyz, length, yaw, roll=0):
    # Size against intrinsic length; rotating a rectangle must not change its real dimensions.
    a.set_actor_rotation(R(), False)
    a.set_actor_scale3d(V(1, 1, 1))
    _, ext = a.get_actor_bounds(False)
    f = length / (max(ext.x, ext.y) * 0.02)
    a.set_actor_scale3d(V(f, f, f))
    a.set_actor_rotation(R(yaw=yaw, roll=roll), False)
    cen, ext = a.get_actor_bounds(False)
    p = a.get_actor_location()
    a.set_actor_location(p + loc(*xyz) - V(cen.x, cen.y, cen.z - ext.z), False, False)
    return a


def update_target(old, row, name, xyz, length, yaw, roll=0, storm=False):
    new = place(library(name, storm), xyz, str(old.get_folder_path()).split("/", 1)[1])
    align_mesh(new, xyz, length, yaw, roll)
    new.set_editor_property("tags", old.tags)
    label = old.get_actor_label()
    ACTORS.destroy_actor(old)
    new.set_actor_label(label)
    lookup[label] = new
    p = new.get_actor_location()
    row["xyz_m"] = [round(v / 100, 3) for v in (p.x, p.y, p.z)]
    row["model"] = name + ("_Storm" if storm else "")
    return new


def add_mesh(key, x, y, z, length, yaw, folder):
    return align_mesh(place(library(key), (x, y, z), folder), (x, y, z), length, yaw)


def backdrop_materials():
    folder = ROOT + "/Materials"
    mat = material(
        "Saltgrass",
        "/Game/MilitaryAirport/Landscape/LandscapeTextures/T_Grass_a_BaseColor_Height",
        "/Game/MilitaryAirport/Landscape/LandscapeTextures/T_Grass_a_Normal",
        tint=(0.82, 1.10, 0.67),
        world_scale=550,
    )
    material(
        "Granite",
        "/Game/MilitaryAirport/Landscape/LandscapeTextures/T_Cliff_BaseColor_Height",
        "/Game/MilitaryAirport/Landscape/LandscapeTextures/T_Cliff_Normal",
        world_scale=300,
    )
    sky_path = folder + "/M_CloudSky"
    if not ED.does_asset_exist(sky_path):
        m = tools.create_asset("M_CloudSky", folder, u.Material, u.MaterialFactoryNew())
        m.set_editor_property("shading_model", u.MaterialShadingModel.MSM_UNLIT)
        m.set_editor_property("two_sided", True)
        m.set_editor_property("is_sky", True)
        camera = lib.create_material_expression(m, u.MaterialExpressionCameraVectorWS)
        inv = lib.create_material_expression(m, u.MaterialExpressionMultiply)
        inv.set_editor_property("const_b", -1)
        assert lib.connect_material_expressions(camera, "", inv, "A")
        sample = lib.create_material_expression(m, u.MaterialExpressionTextureSample)
        sample.set_editor_property("texture", obj(ROOT + "/Library/Sky/HarborSky"))
        assert lib.connect_material_expressions(inv, "", sample, "")
        assert lib.connect_material_property(
            sample, "", u.MaterialProperty.MP_EMISSIVE_COLOR
        )
        lib.recompile_material(m)
        ED.save_loaded_asset(m)
    name = "M_Estuary"
    if not ED.does_asset_exist(folder + "/" + name):
        m = tools.create_asset(name, folder, u.Material, u.MaterialFactoryNew())

        def node(k):
            return lib.create_material_expression(m, k)

        def wire(a, o, b, i):
            assert lib.connect_material_expressions(a, o, b, i)

        def prop(a, o, p):
            assert lib.connect_material_property(a, o, p)

        world = node(u.MaterialExpressionWorldPosition)
        xy = node(u.MaterialExpressionComponentMask)
        xy.set_editor_property("r", True)
        xy.set_editor_property("g", True)
        xy.set_editor_property("b", False)
        xy.set_editor_property("a", False)
        wire(world, "", xy, "")
        normals = []
        for texscale in (670, 2170):
            coord = node(u.MaterialExpressionMultiply)
            coord.set_editor_property("const_b", 1 / texscale)
            wire(xy, "", coord, "A")
            tex = node(u.MaterialExpressionTextureSample)
            tex.set_editor_property(
                "texture",
                obj(
                    "/Engine/Functions/Engine_MaterialFunctions02/ExampleContent/Textures/water_n"
                ),
            )
            tex.set_editor_property(
                "sampler_type", u.MaterialSamplerType.SAMPLERTYPE_NORMAL
            )
            wire(coord, "", tex, "")
            normals.append(tex)
        mix = node(u.MaterialExpressionLinearInterpolate)
        mix.set_editor_property("const_alpha", 0.58)
        wire(normals[0], "", mix, "A")
        wire(normals[1], "", mix, "B")
        prop(mix, "", u.MaterialProperty.MP_NORMAL)
        fresnel = node(u.MaterialExpressionFresnel)
        fresnel.set_editor_property("exponent", 3.0)
        color = node(u.MaterialExpressionLinearInterpolate)
        for i, value in (("A", (0.021, 0.084, 0.074)), ("B", (0.10, 0.18, 0.19))):
            c = node(u.MaterialExpressionConstant3Vector)
            c.set_editor_property("constant", u.LinearColor(*value, 1))
            wire(c, "", color, i)
        wire(fresnel, "", color, "Alpha")
        prop(color, "", u.MaterialProperty.MP_BASE_COLOR)
        r = node(u.MaterialExpressionConstant)
        r.set_editor_property("r", 0.29)
        prop(r, "", u.MaterialProperty.MP_ROUGHNESS)
        lib.recompile_material(m)
        ED.save_loaded_asset(m)
    lookup["Backshore"].static_mesh_component.set_material(0, mat)
    surface(lookup["HarborWater"], "Estuary")
    # Far scenery is visual context beyond the surveyed 2 km square.
    sea = slab("DistantWater", 0, 0, -2.72, 14500, 14500, 0.3, "Estuary", "Backdrop")
    sea.set_actor_enable_collision(False)
    for f in sorted((work / "terrain").glob("*.fbx")):
        folder2 = ROOT + "/Backdrop"
        if not ED.does_asset_exist(folder2 + "/" + f.stem):
            task = u.AssetImportTask()
            opt = u.FbxImportUI()
            opt.set_editor_property("automated_import_should_detect_type", False)
            opt.set_editor_property("import_materials", False)
            opt.set_editor_property("import_textures", False)
            opt.set_editor_property(
                "mesh_type_to_import", u.FBXImportType.FBXIT_STATIC_MESH
            )
            opt.static_mesh_import_data.set_editor_property(
                "auto_generate_collision", False
            )
            for k, v in (
                ("filename", str(f)),
                ("destination_path", folder2),
                ("options", opt),
                ("automated", True),
                ("save", True),
            ):
                task.set_editor_property(k, v)
            tools.import_asset_tasks([task])
    far = place(ROOT + "/Backdrop/FarCoast", (0, 0, 0), "Backdrop", "FarCoastline")
    surface(far, "Saltgrass")
    far.set_actor_enable_collision(False)
    dome = place(
        "/Engine/BasicShapes/Sphere",
        (0, 0, 0),
        "Backdrop",
        "CloudDome",
        scale=(36000, 36000, 36000),
    )
    dome.set_actor_enable_collision(False)
    dome.static_mesh_component.set_editor_property("cast_shadow", False)
    dome.static_mesh_component.set_material(0, obj(sky_path))
    skylight = lookup["MarineSkyLight"].light_component
    skylight.set_editor_property("real_time_capture", False)
    skylight.set_editor_property(
        "source_type", u.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP
    )
    skylight.set_editor_property("cubemap", obj(ROOT + "/Library/Sky/HarborSky"))
    skylight.set_editor_property("lower_hemisphere_is_black", False)
    skylight.set_editor_property("intensity", 1.75)
    sun = lookup["CoastalSun"].light_component
    sun.set_editor_property("intensity", 2.8)
    sun.set_editor_property("light_source_angle", 3.0)


def town():
    house_coords = []
    for row in catalog[:120]:
        i = int(row["id"].split("_")[1])
        a = lookup[row["id"]]
        x, y, z = row["xyz_m"]
        yaw = a.get_actor_rotation().yaw
        kind = row["kind"]
        bad = row["state"] == "damaged"
        if kind == "House":
            j = i - 1
            col = j % 10
            x = (
                (-790, -510, -145, 115, 570)[col // 2]
                + (col % 2) * 37
                + (j // 10 % 3) * 9
                + rng.uniform(-3, 3)
            )
            road = (290, 290, 495, 495, 700, 700, 905)[j // 10]
            sy = (-1, 1, -1, 1, -1, 1, -1)[j // 10]
            y = road + sy * (26 + rng.uniform(-2, 6))
            name = "red-cottage" if (i // 3) % 2 else "village-house"
            length = rng.uniform(12, 17)
            new = update_target(a, row, name, (x, y, 0.03), length, yaw, storm=bad)
            if bad:
                row["visible_evidence"] = "one roof corner locally lifted and twisted"
            slab(
                "Drive_" + row["id"],
                x + length * 0.47,
                (y + road + sy * 7) / 2,
                0.048,
                3.3,
                abs(y - road) - 7,
                0.11,
                "Apron",
                "ResidentialLots",
            )
            house_coords.append((x, y, length))
        elif kind == "Civic":
            name = "harbour-office" if i % 3 else "village-house"
            update_target(
                a, row, name, (x, y, 0.04), rng.uniform(15, 24), yaw, storm=bad
            )
            if bad:
                row["visible_evidence"] = "locally lifted roof edge"
        elif kind == "Shed" and i not in (95, 98):
            name = "warehouse" if i % 3 or bad else "boathouse"
            update_target(
                a, row, name, (x, y, 0.07), rng.uniform(25, 38), yaw, storm=bad
            )
            if bad:
                row["visible_evidence"] = "one roof corner locally lifted and twisted"
        elif kind == "Shop" and not bad and i % 2:
            update_target(
                a, row, "harbour-office", (x, y, 0.07), rng.uniform(14, 19), yaw
            )
    # Preserve space between tree crowns and the building walls after moving the housing clusters.
    for a in list(lookup.values()):
        if str(a.get_folder_path()).endswith("Vegetation"):
            p = a.get_actor_location()
            x, y = p.x / 100, p.y / 100
            if any(
                abs(x - hx) < length * 0.65 + 3 and abs(y - hy) < length * 0.55 + 3
                for hx, hy, length in house_coords
            ):
                ACTORS.destroy_actor(a)


def traffic_and_ships():
    for row in catalog[120:]:
        i = int(row["id"].split("_")[1])
        a = lookup[row["id"]]
        x, y, z = row["xyz_m"]
        yaw = a.get_actor_rotation().yaw
        roll = a.get_actor_rotation().roll
        if row["category"] == "Vessel":
            if row["kind"] == "Fishing":
                name = "skiff" if i % 3 else "fishing-trawler"
            else:
                name = {
                    "Workboat": "tugboat",
                    "Launch": "speedboat",
                    "Coaster": "fishing-trawler",
                }[row["kind"]]
            bad = row["visible_evidence"] == "torn hull plating"
            size = {
                "skiff": 9.0,
                "fishing-trawler": 19.5,
                "tugboat": 14.1,
                "speedboat": 7.1,
            }[name]
            if i in (31, 32):
                size = 28
                row["kind"] = "Offshore trawler"
            if row["state"] == "afloat":
                # Move the two inshore berths per pier beyond the local exposed beach.
                if i <= 25:
                    y = -311 - ((i - 1) % 5) * 22
                draft = {
                    "skiff": 0.56,
                    "fishing-trawler": 1.63,
                    "tugboat": 1.12,
                    "speedboat": 0.42,
                }[name]
                z = (
                    -2.65
                    - draft
                    * size
                    / {
                        "skiff": 9,
                        "fishing-trawler": 19.5,
                        "tugboat": 14.1,
                        "speedboat": 7.1,
                    }[name]
                )
            else:
                z = beach_z(x, y) - 0.09
            update_target(a, row, name, (x, y, z), size, yaw, roll, storm=bad)
            row["visible_evidence"] = (
                "locally indented hull side and bent rail"
                if bad
                else "intact visible hull"
            )
        else:
            kind = row["kind"]
            bad = row["state"] != "intact"
            if kind == "SUV" and not bad:
                continue
            if kind == "Sedan":
                name = "family-sedan" if i % 4 or bad else "pickup-truck"
            elif kind == "Van":
                name = "delivery-van" if i % 3 or bad else "pickup-truck"
            elif kind == "Truck":
                name = "dump-truck" if i % 3 or bad else "semi-truck"
            elif kind == "Utility":
                name = "forklift"
            else:
                name = "tow-truck"
            size = {
                "family-sedan": 4.7,
                "pickup-truck": 5.5,
                "delivery-van": 5.3,
                "dump-truck": 8.3,
                "semi-truck": 12.8,
                "forklift": 3.7,
                "tow-truck": 7.8,
            }[name]
            if name == "forklift":
                x = -459 + ((i - 109) % 4) * 11
                y = -175 + (i - 109) // 4 * 9
                z = 0.17
            else:
                z = 0.18 if i <= 48 else 0.1
            old_roll = roll if row["state"] == "overturned" else 0
            update_target(a, row, name, (x, y, z), size, yaw, old_roll, storm=bad)
            row["kind"] = name
            row["visible_evidence"] = (
                "truck resting on its side with localized body deformation"
                if old_roll
                else "local roof and upper body indentation"
                if bad
                else "intact exterior"
            )


def shore_details():
    for j in range(3):
        add_mesh(
            "harbour-crane",
            -770 + j * 268,
            -247,
            0.18,
            24 + j * 1.8,
            -90 + j * 9,
            "DockEquipment",
        )
    add_mesh("lighthouse", -873, -454, -2.65, 8.3, 2, "Navigation")

    # Rubble mound breakwater, curved in plan. Reuse four rock profiles at varied size and orientation.
    def core(x, y, z, k):
        a = place(
            ROOT + "/Backdrop/Armour" + str(k % 4),
            (x, y, z),
            "Breakwater",
            yaw=rng.uniform(-180, 180),
            scale=(rng.uniform(2.1, 3.2), rng.uniform(2.2, 3.6), rng.uniform(2, 3)),
        )
        surface(a, "Granite")

    for j in range(135):
        t = j / 134
        if t < 0.45:
            x = -870
            y = -230 - t / 0.45 * 222
        else:
            x = -870 + (t - 0.45) / 0.55 * 250
            y = -452 - 15 * math.sin((t - 0.45) / 0.55 * math.pi)
        for side in (-1, 1):
            core(
                x + rng.uniform(-2, 2),
                y + side * rng.uniform(0, 2),
                -2.32 + (j % 7) * 0.05,
                j,
            )
    for j in range(40):
        x = 50 + j * 22.8
        y = shore(x) * 0.42 - 150 * 0.58 + rng.uniform(-5, 5)
        core(x, y, beach_z(x, y) - 0.12, j)
    for j in range(85):
        x = -802 + (j % 17) * 41
        y = -50 + (j // 17) * 6.4
        if j % 4:
            source = (
                layout.PROPS
                + ("Pallet_3_Stack", "Crate_Stack_03", "Barrel_Stack_3")[j % 3]
            )
            # Keep freight on the dock, away from the warehouse foundations.
            layout.fitted(
                source,
                (x, -183 + (j // 17) * 4, 0.18),
                "DockFreight",
                rng.uniform(1.2, 2.2),
                yaw=rng.uniform(-14, 14),
            )
    # Salt-tolerant vegetation above the tidal line, and wet grass inland by the road verges.
    grass = "/Game/MilitaryAirport/Meshes/FoliageMeshes/Grass/SM_Plant_"
    for j in range(210):
        x = rng.uniform(40, 975)
        y = rng.uniform(-156, -134)
        place(
            grass + str((4, 5, 7)[j % 3]),
            (x, y, beach_z(x, y) + 0.01),
            "ShoreGrass",
            yaw=rng.uniform(0, 360),
            scale=(1.8, 1.8, 1.3),
        )
    for j in range(80):
        x = -932 + (j % 20) * 97 + rng.uniform(-12, 12)
        y = 930 + (j // 20) * 15 + rng.uniform(-4, 4)
        add_mesh(
            "silver-birch" if j % 3 else "lombardy-poplar",
            x,
            y,
            0,
            rng.uniform(4.3, 7.4),
            rng.uniform(-180, 180),
            "Shelterbelt",
        )
    # Deliberately few local debris piles: roof loss and car crushing receive different evidence.
    for i in (2, 17, 39):
        p = lookup["Vehicle_%03d" % i].get_actor_location()
        x, y = p.x / 100, p.y / 100
        place(
            local_mesh("Wrack"),
            (x + 2.4, y + 1.2, 0.23),
            "StormDebris",
            yaw=67,
            scale=(0.57, 0.91, 1.2),
        )


def main():
    if not LEVEL.load_level(MAP):
        raise RuntimeError(MAP)
    lookup.update({a.get_actor_label(): a for a in ACTORS.get_all_level_actors()})
    if "CloudDome" in lookup:
        raise RuntimeError("Finished map already exists")
    backdrop_materials()
    town()
    traffic_and_ships()
    shore_details()
    # Reconcile catalog positions for all actor roots after refinement.
    for row in catalog:
        p = lookup[row["id"]].get_actor_location()
        row["xyz_m"] = [round(v / 100, 3) for v in (p.x, p.y, p.z)]
    assert LEVEL.save_current_level()
    (work / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf8")
    u.log("HARBOR_FINAL_LEVEL_SAVED")


main()
