"""Run with UE 5.2's PythonScript commandlet next to the Blender meshes/ directory."""

from pathlib import Path
import unreal as u

ROOT = "/Game/CoastalHarbor"
ED = u.EditorAssetLibrary
tools = u.AssetToolsHelpers.get_asset_tools()
lib = u.MaterialEditingLibrary
tile = "/Game/MilitaryAirport/Textures/Tileable/T_"
land = "/Game/MilitaryAirport/Landscape/LandscapeTextures/T_"


def material(
    name,
    color_map=None,
    normal_map=None,
    tint=None,
    rough=0.65,
    metal=0,
    vertex=False,
    world_scale=None,
):
    path = ROOT + "/Materials/M_" + name
    if ED.does_asset_exist(path):
        return ED.load_asset(path)
    mat = tools.create_asset(
        "M_" + name, ROOT + "/Materials", u.Material, u.MaterialFactoryNew()
    )

    def node(cls, x, y):
        return lib.create_material_expression(mat, cls, x, y)

    def wire(a, output, b, inp):
        if not lib.connect_material_expressions(a, output, b, inp):
            raise RuntimeError((name, output, inp))

    def prop(a, out, mp):
        if not lib.connect_material_property(a, out, mp):
            raise RuntimeError((name, str(mp)))

    coord = None
    if world_scale:
        pos = node(u.MaterialExpressionWorldPosition, -1050, 300)
        mask = node(u.MaterialExpressionComponentMask, -880, 300)
        for key, value in (("r", True), ("g", True), ("b", False), ("a", False)):
            mask.set_editor_property(key, value)
        wire(pos, "", mask, "")
        coord = node(u.MaterialExpressionMultiply, -680, 300)
        coord.set_editor_property("const_b", 1 / world_scale)
        wire(mask, "", coord, "A")

    def sample(path, normal=False):
        tex = u.load_asset(path)
        if not tex:
            raise RuntimeError(path)
        n = node(u.MaterialExpressionTextureSample, -420, 210 if normal else -130)
        n.set_editor_property("texture", tex)
        if normal:
            n.set_editor_property(
                "sampler_type", u.MaterialSamplerType.SAMPLERTYPE_NORMAL
            )
        if coord:
            wire(coord, "", n, "")
        return n

    base = None
    if color_map:
        base = sample(color_map)
    if tint or vertex:
        if vertex:
            c = node(u.MaterialExpressionVertexColor, -460, 0)
        else:
            c = node(u.MaterialExpressionConstant3Vector, -460, 0)
            c.set_editor_property("constant", u.LinearColor(*tint, 1))
        if base:
            mult = node(u.MaterialExpressionMultiply, -140, 0)
            wire(c, "", mult, "A")
            wire(base, "", mult, "B")
            prop(mult, "", u.MaterialProperty.MP_BASE_COLOR)
        else:
            prop(c, "", u.MaterialProperty.MP_BASE_COLOR)
    elif base:
        prop(base, "", u.MaterialProperty.MP_BASE_COLOR)
    if normal_map:
        prop(sample(normal_map, True), "", u.MaterialProperty.MP_NORMAL)
    for value, mp, yy in (
        (rough, u.MaterialProperty.MP_ROUGHNESS, 420),
        (metal, u.MaterialProperty.MP_METALLIC, 510),
    ):
        n = node(u.MaterialExpressionConstant, -130, yy)
        n.set_editor_property("r", value)
        prop(n, "", mp)
    mat.set_editor_property("two_sided", True)
    lib.recompile_material(mat)
    if not ED.save_loaded_asset(mat):
        raise RuntimeError(path)
    return mat


def main():
    for name, tex, rough, metal in (
        ("Paint", None, 0.39, 0.24),
        ("Timber", "Planks", 0.76, 0),
        ("Roof", "CorrugatedPanel", 0.53, 0.35),
        ("Glass", None, 0.13, 0.48),
        ("Rubber", "Rubber", 0.86, 0),
        ("Plaster", "Plaster", 0.81, 0),
    ):
        material(
            name,
            tile + tex + "_BC" if tex else None,
            tile + tex + "_N" if tex else None,
            rough=rough,
            metal=metal,
            vertex=True,
        )
    material(
        "Ground",
        land + "Grass_b_BaseColor_Height",
        land + "Grass_b_Normal",
        world_scale=830,
    )
    material(
        "Mud",
        land + "Puddle_Sand_BaseColor_Height",
        land + "Puddle_Sand_Normal",
        rough=0.40,
        world_scale=670,
    )
    for name, tex, scale in (
        ("Road", "Asphalt", 440),
        ("Apron", "ConcreteFloor", 530),
        ("Pier", "Planks", 310),
    ):
        material(name, tile + tex + "_BC", tile + tex + "_N", world_scale=scale)
    material(
        "Water",
        normal_map="/Engine/Functions/Engine_MaterialFunctions02/ExampleContent/Textures/water_n",
        tint=(0.023, 0.135, 0.153),
        rough=0.20,
        metal=0.24,
        world_scale=1250,
    )
    material("Stripe", tint=(0.66, 0.65, 0.51))
    material("Oxide", tint=(0.29, 0.112, 0.052))
    tasks = []
    for file in sorted((Path(__file__).parent / "meshes").glob("*.fbx")):
        if ED.does_asset_exist(ROOT + "/Meshes/" + file.stem):
            continue
        task = u.AssetImportTask()
        options = u.FbxImportUI()
        options.set_editor_property("automated_import_should_detect_type", False)
        options.set_editor_property(
            "mesh_type_to_import", u.FBXImportType.FBXIT_STATIC_MESH
        )
        options.set_editor_property("import_materials", False)
        options.set_editor_property("import_textures", False)
        options.set_editor_property("import_as_skeletal", False)
        data = options.static_mesh_import_data
        data.set_editor_property("combine_meshes", True)
        data.set_editor_property(
            "vertex_color_import_option", u.VertexColorImportOption.REPLACE
        )
        data.set_editor_property("auto_generate_collision", False)
        for key, val in (
            ("filename", str(file)),
            ("destination_path", ROOT + "/Meshes"),
            ("automated", True),
            ("save", True),
            ("options", options),
        ):
            task.set_editor_property(key, val)
        tasks.append(task)
    tools.import_asset_tasks(tasks)
    for file in sorted((Path(__file__).parent / "meshes").glob("*.fbx")):
        mesh = ED.load_asset(ROOT + "/Meshes/" + file.stem)
        if not mesh:
            raise RuntimeError(str(file))
        slots = []
        for i, sm in enumerate(mesh.static_materials):
            slot = str(sm.material_slot_name)
            slots.append(slot)
            mat = ED.load_asset(ROOT + "/Materials/M_" + slot)
            if mat is None:
                raise RuntimeError((file.stem, slot))
            mesh.set_material(i, mat)
        mesh.get_editor_property("body_setup").set_editor_property(
            "collision_trace_flag", u.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
        )
        ED.save_loaded_asset(mesh, False)
        if file.stem in ("House0", "Fishing0", "Backshore", "Foreshore"):
            u.log("HARBOR_MESH %s %s %s" % (file.stem, mesh.get_bounding_box(), slots))
    u.log("HARBOR_IMPORT_COMPLETE")


if __name__ == "__main__":
    main()
