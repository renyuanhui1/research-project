"""Populate the saved harbor's existing street grid. Run after harbor_landscape in UE 5.2."""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import unreal as u

work = Path(__file__).resolve().parent
sys.path.insert(0, str(work))
from harbor_import import material
from harbor_level import (
    ROOT,
    MAP,
    LEVEL,
    ACTORS,
    ED,
    PROPS,
    R,
    V,
    place,
    slab,
    fitted,
    obj,
)

folder = "StreetFrontages"
mesh_root = ROOT + "/StreetArchitecture/"
counts = Counter()
rng = random.Random(11409)


def import_kit():
    src = (
        Path(u.Paths.convert_relative_path_to_full(u.Paths.project_dir()))
        / "SourceArt/CoastalHarbor/streets"
    )
    files = sorted(src.glob("*.fbx"))
    assert len(files) == 10, src
    tasks = []
    for f in files:
        if ED.does_asset_exist(mesh_root + f.stem):
            continue
        t = u.AssetImportTask()
        opt = u.FbxImportUI()
        opt.automated_import_should_detect_type = False
        opt.mesh_type_to_import = u.FBXImportType.FBXIT_STATIC_MESH
        opt.import_materials = opt.import_textures = opt.import_as_skeletal = False
        opt.static_mesh_import_data.combine_meshes = True
        opt.static_mesh_import_data.auto_generate_collision = False
        opt.static_mesh_import_data.vertex_color_import_option = (
            u.VertexColorImportOption.REPLACE
        )
        t.filename, t.destination_path, t.automated, t.save, t.options = (
            str(f),
            mesh_root.rstrip("/"),
            True,
            True,
            opt,
        )
        tasks.append(t)
    if tasks:
        u.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    for f in files:
        mesh = obj(mesh_root + f.stem)
        for i, slot in enumerate(mesh.static_materials):
            mesh.set_material(
                i, obj(ROOT + "/Materials/M_" + str(slot.material_slot_name))
            )
        mesh.get_editor_property("body_setup").set_editor_property(
            "collision_trace_flag", u.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
        )
        ED.save_loaded_asset(mesh, False)


def box(a):
    c, e = a.get_actor_bounds(False)
    return c.x / 100, c.y / 100, e.x / 100, e.y / 100


def exceeds(x, y, w, d):
    return any(abs(x - a) < w + c and abs(y - b) < d + e for a, b, c, e in occupied)


def axis(mesh):
    b = mesh.get_bounding_box()
    if b.max.x - b.min.x < b.max.y - b.min.y:
        return 0 if b.max.x > -b.min.x else 180
    return 90 if b.max.y > -b.min.y else -90


def building(key, x, y, side, label=None):
    mesh = obj(mesh_root + key)
    # Porch and canopy extend furthest from the street facade along the short axis.
    yaw = -90 * side - axis(mesh)
    counts[key] += 1
    return place(mesh, (x, y, 0.06), folder, label, yaw=yaw)


def paving(label, x, y, top, w, d):
    return slab(label, x, y, top, w, d, 0.15, "Apron", folder)


def garden(x, y, road, half, key):
    side = 1 if y > road else -1
    kerb = road + side * 9.4
    entry_depth = half + (0.1 if key == "Existing" else 1.7)
    entrance = y - side * entry_depth
    units = {
        "HouseHarborTerrace": 3,
        "HouseDormerPair": 2,
        "HouseHippedCottage": 1,
    }.get(key, 0)
    offsets = [-side * ((i - (units - 1) / 2) * 6.1 + 1.55) for i in range(units)] or [
        0
    ]
    path_x = x + min(offsets, key=abs)
    paving(
        "GardenWalk",
        path_x,
        (entrance + kerb) / 2,
        0.069,
        1.55,
        abs(entrance - kerb) + 1.3,
    )
    if units > 1:
        left, right = min(offsets), max(offsets)
        paving(
            "PorchWalk",
            x + (left + right) / 2,
            entrance,
            0.074,
            right - left + 1.5,
            1.43,
        )
    rail = obj(mesh_root + "GardenRail")
    b = rail.get_bounding_box()
    yaw = 0 if b.max.x - b.min.x > b.max.y - b.min.y else 90
    place(rail, (path_x, kerb + side * 2, 0.04), folder, yaw=yaw)
    garden_plots.append((x, y, side, half, yaw))
    counts["garden_frontages"] += 1


def backyards():
    rng = random.Random(3315)
    trees = ROOT + "/Library/"
    for j, (x, y, side, depth, yaw) in enumerate(garden_plots):
        # Vary rear lot depth and preserve access down one side of each dwelling.
        rear = y + side * (depth + rng.uniform(9.8, 15.6))
        if not exceeds(x, rear, 7.8, 1):
            place(obj(mesh_root + "GardenRail"), (x, rear, 0.04), folder, yaw=yaw + 180)
            counts["rear_boundaries"] += 1
            length = abs(rear - y) - depth + 2
            edge_x = x + 8.6
            edge_y = rear - side * length / 2
            if not exceeds(edge_x, edge_y, 0.24, length / 2):
                slab(
                    "GardenWall",
                    edge_x,
                    edge_y,
                    0.67,
                    0.24,
                    length,
                    0.67,
                    "Apron",
                    folder,
                )
            if j % 3 == 0:
                for dx in (-4.8, -3.1, 3.5, 5.3):
                    if not exceeds(x + dx, rear - side * 0.60, 0.6, 0.6):
                        place(
                            "/Game/MilitaryAirport/Meshes/FoliageMeshes/Grass/SM_Plant_"
                            + str(4 + (j % 2)),
                            (x + dx, rear - side * 0.60, 0.03),
                            folder,
                            yaw=rng.uniform(-180, 180),
                            scale=(1.6, 1.6, 1.3),
                        )
        xx = x + rng.choice((-5.6, 5.6))
        yy = rear - side * 3.7
        if j % 3 != 1 and not exceeds(xx, yy, 2.9, 2.8):
            a = place(
                obj(mesh_root + "GardenWorkshop"),
                (xx, yy, 0.025),
                folder,
                yaw=-side * 90 - axis(obj(mesh_root + "GardenWorkshop")),
            )
            occupied.append(box(a))
            counts["garden_workshops"] += 1
        for k in range(1 + (j % 4 == 0)):
            tx = x + rng.choice((-1, 1)) * rng.uniform(4.8, 8.8)
            ty = rear + side * rng.uniform(1.2, 5.5)
            if exceeds(tx, ty, 3.2, 3.0):
                continue
            species = "silver_birch" if (j + k) % 4 else "lombardy_poplar"
            tree = fitted(
                trees + species + "/" + species,
                (tx, ty, 0.01),
                folder,
                rng.uniform(3.4, 5.9),
                yaw=rng.uniform(-180, 180),
            )
            occupied.append(box(tree))
            counts["garden_trees"] += 1


def sidewalks():
    # South pavements terminate at cross streets, with a curb cut at every junction.
    for road in (85, 290, 495, 700, 905):
        for start, end in [(-884, -315), (-285, 285), (315, 884)]:
            paving(
                "SouthFootway", (start + end) / 2, road - 7.78, 0.115, end - start, 3.06
            )


def housing():
    kinds = ("HouseHarborTerrace", "HouseDormerPair", "HouseHippedCottage")
    for row, (road, side) in enumerate(
        ((290, -1), (290, 1), (495, -1), (495, 1), (700, -1), (700, 1), (905, -1))
    ):
        for start, end in (
            (-874, -652),
            (-598, -382),
            (-222, -48),
            (48, 213),
            (474, 682),
        ):
            x = start + 5 * rng.random()
            while x < end:
                key = rng.choices(kinds, weights=(2, 3, 2))[0]
                y = road + side * (19 + rng.uniform(-2.5, 3))
                w = {
                    "HouseHarborTerrace": 9.7,
                    "HouseDormerPair": 6.7,
                    "HouseHippedCottage": 4.6,
                }[key]
                if not exceeds(x, y, w + 2, 7.3):
                    a = building(key, x, y, side)
                    occupied.append(box(a))
                    garden(x, y, road, 4.15, key)
                x += rng.uniform(22, 27)
    for key, a in targets.items():
        if key.startswith("Building_") and int(key[-3:]) <= 70:
            x, y, w, d = box(a)
            road = min((290, 495, 700, 905), key=lambda r: abs(y - r))
            garden(x, y, road, d, "Existing")


def shopping():
    for i in range(71, 95):
        label = "Building_%03d" % i
        a = targets[label]
        x, y, w, d = box(a)
        side = 1 if y > 85 else -1
        if i in (71, 85):
            key = "ShopFishMarket"
            mesh = obj(mesh_root + key)
            comp = a.static_mesh_component
            comp.set_static_mesh(mesh)
            comp.set_editor_property("override_materials", [])
            a.set_actor_rotation(R(yaw=-90 * side - axis(mesh)), False)
            a.set_actor_scale3d(V(1, 1, 1))
            rows[label]["model"] = key
            rows[label][
                "visible_evidence"
            ] = "one end of striped shop awning torn loose and hanging from facade"
            x, y, w, d = box(a)
            counts["damaged_market_fronts"] += 1
        curb = 85 + side * 9.35
        far = y + side * min(d, 5.0)
        paving(
            "ShopForecourt",
            x,
            (curb + far) / 2,
            0.084,
            2 * w + 3,
            abs(far - curb) + 0.7,
        )
        for dx in ((-25, 24) if i % 3 else (-25, 24, 46)):
            xx = x + dx
            if any(abs(xx - r) < 15 for r in (-900, -300, 300, 900)) or exceeds(
                xx, y, 7.8, 6.5
            ):
                continue
            key = (
                "ShopChandlery",
                "HouseDormerPair",
                "ShopBakery",
                "HouseHarborTerrace",
                "HouseHippedCottage",
            )[(i + int(dx > 0) * 2) % 5]
            if key == "HouseHarborTerrace" and exceeds(xx, y, 10.2, 7):
                continue
            a = building(key, xx, y, side)
            occupied.append(box(a))
            if key.startswith("House"):
                garden(xx, y, 85, 4.15, key)
            else:
                paving(
                    "ShopForecourt", xx, (y + curb) / 2, 0.081, 16, abs(y - curb) + 1
                )
                if key == "ShopBakery":
                    place(
                        obj(mesh_root + "BakeryTables"),
                        (xx + 2, y - side * 8, 0.09),
                        folder,
                        yaw=90,
                    )
                    counts["cafe_terraces"] += 1
        if i in (71, 85):
            place(
                obj(mesh_root + "QuayFishStall"),
                (x - 3.4, y - side * 7.0, 0.10),
                folder,
            )
            counts["fish_displays"] += 1


def meadow_surface(dest):
    name = "M_MixedCoastalGrass"
    path = ROOT + "/Materials/" + name
    if ED.does_asset_exist(path):
        return obj(path)
    mat = u.AssetToolsHelpers.get_asset_tools().create_asset(
        name, ROOT + "/Materials", u.Material, u.MaterialFactoryNew()
    )
    lib = u.MaterialEditingLibrary

    def node(kind):
        return lib.create_material_expression(mat, kind)

    def connect(source, output, target, slot):
        assert lib.connect_material_expressions(source, output, target, slot)

    xy = node(u.MaterialExpressionComponentMask)
    for k in "rgba":
        xy.set_editor_property(k, k in "rg")
    connect(node(u.MaterialExpressionWorldPosition), "", xy, "")

    def sample(asset, period, normal=False):
        scale = node(u.MaterialExpressionMultiply)
        scale.set_editor_property("const_b", 1.0 / period)
        connect(xy, "", scale, "A")
        t = node(u.MaterialExpressionTextureSample)
        t.set_editor_property("texture", obj(asset))
        if normal:
            t.set_editor_property(
                "sampler_type", u.MaterialSamplerType.SAMPLERTYPE_NORMAL
            )
        connect(scale, "", t, "")
        return t

    # Two photographic footprints break up repeated grass patches. The broad meadow
    # mask changes their mixture continuously across the full length of the town.
    scan = dest + "/MaritimeSward_BC"
    mask = sample(ROOT + "/Backdrop/MeadowMask", 48600)
    fine = sample(scan, 1910)
    broad = sample(scan, 14650)
    blend = node(u.MaterialExpressionLinearInterpolate)
    connect(fine, "RGB", blend, "A")
    connect(broad, "RGB", blend, "B")
    connect(mask, "R", blend, "Alpha")
    pigment = node(u.MaterialExpressionConstant3Vector)
    pigment.set_editor_property("constant", u.LinearColor(0.072, 0.096, 0.056, 1))
    grass = node(u.MaterialExpressionLinearInterpolate)
    grass.set_editor_property("const_alpha", 0.48)
    connect(blend, "", grass, "A")
    connect(pigment, "", grass, "B")
    assert lib.connect_material_property(grass, "", u.MaterialProperty.MP_BASE_COLOR)
    assert lib.connect_material_property(
        sample(dest + "/MaritimeSward_N", 1910, True),
        "RGB",
        u.MaterialProperty.MP_NORMAL,
    )
    rough = node(u.MaterialExpressionConstant)
    rough.set_editor_property("r", 0.91)
    assert lib.connect_material_property(rough, "", u.MaterialProperty.MP_ROUGHNESS)
    lib.recompile_material(mat)
    ED.save_loaded_asset(mat, False)
    return mat


def freight():
    scan_path = (
        Path(u.Paths.convert_relative_path_to_full(u.Paths.project_dir()))
        / "SourceArt/CoastalHarbor/streets/Ground"
    )
    dest = mesh_root.rstrip("/") + "/Ground"
    for f in sorted(scan_path.glob("*.jpg")):
        if not ED.does_asset_exist(dest + "/" + f.stem):
            t = u.AssetImportTask()
            t.filename, t.destination_path, t.automated, t.save = (
                str(f),
                dest,
                True,
                True,
            )
            u.AssetToolsHelpers.get_asset_tools().import_asset_tasks([t])
        tex = obj(dest + "/" + f.stem)
        if f.stem.endswith("_N"):
            tex.set_editor_property(
                "compression_settings", u.TextureCompressionSettings.TC_NORMALMAP
            )
            tex.set_editor_property("srgb", False)
            ED.save_loaded_asset(tex, False)
    # Warm aggregate in the sun reads neutral when mixed with local silty fill.
    gravel = material(
        "CompactedQuayGravel",
        dest + "/YardStone_BC",
        dest + "/YardStone_N",
        tint=(0.57, 0.68, 0.75),
        rough=0.91,
        world_scale=390,
    )
    sward = meadow_surface(dest)
    entries = {a.get_actor_label(): a for a in ACTORS.get_all_level_actors()}
    entries["Backshore"].static_mesh_component.set_material(0, sward)
    for key in ("ServiceYard_-445", "ServiceYard_460", "ServiceYard_-520"):
        entries[key].static_mesh_component.set_material(0, gravel)
    for i in range(95, 111):
        a = targets["Building_%03d" % i]
        x, y, w, d = box(a)
        bay_x = x + w + 6
        paving("LoadingHardstand", bay_x, (y - 114) / 2, 0.086, 10, abs(y + 114) + 1)
        paving("LoadingCourt", x, y, 0.071, 2 * w + 13, 2 * d + 12)
        for j, (kind, span) in enumerate(
            (("Crate_Stack_02", 2.4), ("Pallet_3_Stack", 2), ("Crate_Stack_04", 2.5))
        ):
            yy = y + 4 - j * 4.7
            if not exceeds(bay_x, yy, 1.6, 1.7):
                fitted(
                    PROPS + kind,
                    (bay_x, yy, 0.09),
                    folder,
                    span,
                    yaw=rng.uniform(-6, 6),
                )
                counts["freight_props"] += 1
        for k in (-1, 1):
            xx = bay_x + 4 * k
            if not exceeds(xx, y - 11, 0.5, 0.5):
                fitted(PROPS + "Cone_2_Bollard", (xx, y - 11, 0.09), folder, 0.42)
        if i % 3 != 0 and not exceeds(x, y + d + 9, 6, 3):
            a = fitted(
                PROPS + "Container",
                (x, y + d + 9, 0.085),
                folder,
                11.7,
                yaw=3 * (i % 3),
            )
            occupied.append(box(a))
            counts["shipping_containers"] += 1
            ax, ay, aw, ad = box(a)
            for k in range(1, 2 + (i % 3 == 1)):
                yy = ay + k * 3.25
                if not exceeds(ax, yy, aw + 0.3, 1.45):
                    c = fitted(
                        PROPS + ("Container_2" if k % 2 else "Container"),
                        (ax, yy, 0.08),
                        folder,
                        11.7,
                        yaw=0,
                    )
                    occupied.append(box(c))
                    counts["shipping_containers"] += 1
            if i % 2 == 0:
                center, extent = a.get_actor_bounds(False)
                fitted(
                    PROPS + "Container_2",
                    (ax, ay, (center.z + extent.z) / 100),
                    folder,
                    11.65,
                    yaw=a.get_actor_rotation().yaw,
                )
                counts["shipping_containers"] += 1


def run():
    global targets, occupied, rows, garden_plots
    garden_plots = []
    assert LEVEL.load_level(MAP)
    actors = ACTORS.get_all_level_actors()
    targets = {
        a.get_actor_label(): a
        for a in actors
        if a.get_actor_label().startswith(("Building_", "Vehicle_", "Vessel_"))
    }
    rows = {
        r["id"]: r
        for r in json.loads((work / "catalog.json").read_text(encoding="utf8"))
    }
    assert len(targets) == len(rows) == 280
    for a in actors:
        if str(a.get_folder_path()) == "Harbor/" + folder:
            ACTORS.destroy_actor(a)
    occupied = [box(a) for a in targets.values()]
    for a in ACTORS.get_all_level_actors():
        if str(a.get_folder_path()).split("/")[-1] in (
            "Vegetation",
            "StreetFurniture",
            "Lights",
        ):
            occupied.append(box(a))
    camfile = work / "street-cameras.json"
    if not camfile.exists():
        c, e = targets["Building_071"].get_actor_bounds(False)
        center = [v / 100 for v in (c.x, c.y, c.z)]
        camfile.write_text(
            json.dumps(
                [
                    [
                        "07_roadside",
                        [v + d for v, d in zip(center, (7, 20, 6))],
                        center,
                    ],
                    ["11_market_street", [-811, 81, 4], [-836, 57, 3.3]],
                    ["12_street_level", [-843, 282.3, 2.25], [-817, 270, 2.95]],
                ]
            )
        )
    import_kit()
    sidewalks()
    housing()
    shopping()
    freight()
    backyards()
    assert sum(counts[k] for k in counts if k.startswith("House")) > 70, counts
    assert LEVEL.save_current_level()
    (work / "street-build.json").write_text(json.dumps(counts, indent=2))
    (work / "catalog.json").write_text(
        json.dumps(list(rows.values()), indent=2), encoding="utf8"
    )
    u.log("STREETS_SAVED " + str(counts))


if __name__ == "__main__":
    run()
