"""Blend coastal soils at two scales and complete residential and mooring-area dressing."""

from pathlib import Path
import json
import sys
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
    local_mesh,
    PROPS,
    fitted,
)

tools = u.AssetToolsHelpers.get_asset_tools()
lib = u.MaterialEditingLibrary


def meadow():
    folder = ROOT + "/Materials"
    name = "M_CoastalMeadow"
    if ED.does_asset_exist(folder + "/" + name):
        return obj(folder + "/" + name)
    if not ED.does_asset_exist(ROOT + "/Backdrop/MeadowMask"):
        t = u.AssetImportTask()
        for k, v in (
            ("filename", str(work / "terrain/MeadowMask.png")),
            ("destination_path", ROOT + "/Backdrop"),
            ("automated", True),
            ("save", True),
        ):
            t.set_editor_property(k, v)
        tools.import_asset_tasks([t])
    m = tools.create_asset(name, folder, u.Material, u.MaterialFactoryNew())

    def node(k):
        return lib.create_material_expression(m, k)

    def wire(a, o, b, i):
        assert lib.connect_material_expressions(a, o, b, i)

    def prop(a, o, mp):
        assert lib.connect_material_property(a, o, mp)

    xy = node(u.MaterialExpressionComponentMask)
    for c, v in (("r", True), ("g", True), ("b", False), ("a", False)):
        xy.set_editor_property(c, v)
    wire(node(u.MaterialExpressionWorldPosition), "", xy, "")

    def sample(path, scale, normal=False):
        uv = node(u.MaterialExpressionMultiply)
        uv.set_editor_property("const_b", 1 / scale)
        wire(xy, "", uv, "A")
        tex = node(u.MaterialExpressionTextureSample)
        tex.set_editor_property("texture", obj(path))
        if normal:
            tex.set_editor_property(
                "sampler_type", u.MaterialSamplerType.SAMPLERTYPE_NORMAL
            )
        wire(uv, "", tex, "")
        return tex

    macro = sample(ROOT + "/Backdrop/MeadowMask", 56600)
    palette = node(u.MaterialExpressionLinearInterpolate)
    for slot, color in (("A", (0.077, 0.099, 0.050)), ("B", (0.187, 0.164, 0.102))):
        n = node(u.MaterialExpressionConstant3Vector)
        n.set_editor_property("constant", u.LinearColor(*color, 1))
        wire(n, "", palette, slot)
    wire(macro, "R", palette, "Alpha")
    land = "/Game/MilitaryAirport/Landscape/LandscapeTextures/T_"
    micro = sample(land + "Grass_b_BaseColor_Height", 355)
    gain = node(u.MaterialExpressionAdd)
    gain.set_editor_property("const_b", 0.62)
    wire(micro, "G", gain, "A")
    col = node(u.MaterialExpressionMultiply)
    wire(palette, "", col, "A")
    wire(gain, "", col, "B")
    prop(col, "", u.MaterialProperty.MP_BASE_COLOR)
    prop(sample(land + "Grass_b_Normal", 355, True), "", u.MaterialProperty.MP_NORMAL)
    lib.recompile_material(m)
    ED.save_loaded_asset(m)
    return m


def main():
    assert LEVEL.load_level(MAP)
    actors = {a.get_actor_label(): a for a in ACTORS.get_all_level_actors()}
    if "QuayParking_00" in actors:
        raise RuntimeError("Landscape already finished")
    mat = meadow()
    for label in ("Backshore", "FarCoastline"):
        actors[label].static_mesh_component.set_material(0, mat)
    # Lift distant relief so its troughs stay above the sea beyond the inland boundary.
    far = actors["FarCoastline"]
    far.set_actor_scale3d(V(1, 1, 0.72))
    far.set_actor_location(loc(0, 0, 8), False, False)
    # Remove close intrusions from the incidental rock scatter around grounded hulls.
    rows = json.loads((work / "catalog.json").read_text(encoding="utf8"))
    hulls = [
        actors[r["id"]].get_actor_bounds(False) for r in rows if r["state"] == "aground"
    ]
    for a in list(actors.values()):
        if str(a.get_folder_path()).endswith("Breakwater"):
            c, e = a.get_actor_bounds(False)
            if any(
                abs(c.x - p.x) < e.x + d.x + 55 and abs(c.y - p.y) < e.y + d.y + 55
                for p, d in hulls
            ):
                ACTORS.destroy_actor(a)
    rng = random.Random(219)
    for j in range(13):
        x = -729.55 + j * 7.1
        for k in range(4):
            slab(
                "QuayParking_%02d" % (j * 4 + k),
                x,
                -203 + k * 9,
                0.18 + 0.012,
                0.13,
                5.5,
                0.018,
                "Stripe",
                "Parking",
            )
    for row in rows[:120]:
        a = actors[row["id"]]
        c, e = a.get_actor_bounds(False)
        x, y = c.x / 100, c.y / 100
        if row["kind"] == "Shop" and e.z * 2 / 100 > 16:
            # Keep the coastal high street at the scale of the adjacent houses.
            size = a.get_actor_scale3d()
            p = a.get_actor_location()
            a.set_actor_scale3d(V(size.x, size.y, size.z * 0.48))
            cc, ee = a.get_actor_bounds(False)
            p.z += 4 - cc.z + ee.z
            a.set_actor_location(p, False, False)
        if row["kind"] == "House":
            i = int(row["id"].split("_")[1])
            side = (-1, 1, -1, 1, -1, 1, -1)[(i - 1) // 10]
            if i % 3:
                place(
                    local_mesh("Wrack"),
                    (x - 9, y + side * 7, 0.025),
                    "GardenEdges",
                    yaw=rng.uniform(40, 80),
                    scale=(0.25, 0.38, 0.5),
                )
            # Small trees beside the houses give useful scale and partial occlusion.
            if i % 2 == 0:
                fitted(
                    "/Game/ModernCityBundle/Meshes/StreetElements/SM_Tree_1",
                    (x + e.x / 100 + 4.5, y + side * 5, 0),
                    "GardenTrees",
                    rng.uniform(3.8, 6.2),
                    yaw=rng.uniform(0, 360),
                )
    for i in (1, 6, 13):
        x = -776 + i * 32
        for j, kind in enumerate(("Crate_1", "Crate_2", "Pallet_2")):
            fitted(
                PROPS + kind,
                (x + j * 1.7, -244, 0.17),
                "LandingGear",
                1.4,
                yaw=rng.uniform(-8, 8),
            )
    assert LEVEL.save_current_level()
    u.log("HARBOR_LANDSCAPE_SAVED")


main()
