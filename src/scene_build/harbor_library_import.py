"""Import textured civilian models and their local storm-damage variants into UE5."""

import json
from pathlib import Path
import unreal as u

base = "/Game/CoastalHarbor/Library"
files = Path(__file__).parent / "library"
lib = u.MaterialEditingLibrary
tools = u.AssetToolsHelpers.get_asset_tools()
ed = u.EditorAssetLibrary
masters = {}
changed = []


def asset(name, folder, cls, factory):
    result = (
        u.load_asset(folder + "/" + name)
        if ed.does_asset_exist(folder + "/" + name)
        else None
    )
    if result is None:
        result = tools.create_asset(name, folder, cls, factory)
    if not result:
        raise RuntimeError((folder, name))
    return result


def importer(path, folder, mesh=False):
    task = u.AssetImportTask()
    for key, value in (
        ("filename", str(path)),
        ("destination_path", folder),
        ("replace_existing", False),
        ("automated", True),
        ("save", False),
    ):
        task.set_editor_property(key, value)
    if mesh:
        options = u.FbxImportUI()
        options.set_editor_property("automated_import_should_detect_type", False)
        options.set_editor_property(
            "mesh_type_to_import", u.FBXImportType.FBXIT_STATIC_MESH
        )
        options.set_editor_property("import_materials", False)
        options.set_editor_property("import_textures", False)
        options.set_editor_property("import_as_skeletal", False)
        options.static_mesh_import_data.set_editor_property("combine_meshes", True)
        options.static_mesh_import_data.set_editor_property(
            "auto_generate_collision", False
        )
        task.set_editor_property("options", options)
    return task


def load_files(paths, folder):
    queue = [
        importer(p, folder, p.suffix == ".fbx")
        for p in paths
        if not ed.does_asset_exist(folder + "/" + p.stem)
    ]
    if queue:
        tools.import_asset_tasks(queue)
    assets = {p.stem: u.load_asset(folder + "/" + p.stem) for p in paths}
    if not all(assets.values()):
        raise RuntimeError("Missing imported file in " + folder)
    return assets


def master(channels, masked):
    key = "PBR_" + "".join(k[:1] for k in channels) + ("Leaves" if masked else "Solid")
    if key in masters:
        return masters[key]
    folder = base + "/Materials"
    if ed.does_asset_exist(folder + "/" + key):
        masters[key] = u.load_asset(folder + "/" + key)
        return masters[key]
    m = asset(key, folder, u.Material, u.MaterialFactoryNew())
    m.set_editor_property("two_sided", True)
    if masked:
        m.set_editor_property("blend_mode", u.BlendMode.BLEND_MASKED)

    def expr(cls):
        return lib.create_material_expression(m, cls)

    def wire(a, out, b, inp):
        if not lib.connect_material_expressions(a, out, b, inp):
            raise RuntimeError((key, out, inp))

    def prop(a, out, mp):
        if not lib.connect_material_property(a, out, mp):
            raise RuntimeError((key, out, str(mp)))

    def param(cls, name, val):
        node = expr(cls)
        node.set_editor_property("parameter_name", name)
        node.set_editor_property("default_value", val)
        return node

    def mul(a, out, b):
        n = expr(u.MaterialExpressionMultiply)
        wire(a, out, n, "A")
        wire(b, "", n, "B")
        return n

    coord = mul(
        expr(u.MaterialExpressionTextureCoordinate),
        "",
        param(u.MaterialExpressionScalarParameter, "UVScale", 1.0),
    )
    samples = {}
    for ch, tex in channels.items():
        sampler = expr(u.MaterialExpressionTextureSampleParameter2D)
        sampler.set_editor_property("parameter_name", ch)
        sampler.set_editor_property("texture", tex)
        if ch == "Normal":
            sampler.set_editor_property(
                "sampler_type", u.MaterialSamplerType.SAMPLERTYPE_NORMAL
            )
        if ch == "Packed":
            sampler.set_editor_property(
                "sampler_type", u.MaterialSamplerType.SAMPLERTYPE_MASKS
            )
        wire(coord, "", sampler, "")
        samples[ch] = sampler
    tint = param(u.MaterialExpressionVectorParameter, "Tint", u.LinearColor(1, 1, 1, 1))
    color = mul(samples["Diffuse"], "", tint) if "Diffuse" in samples else tint
    prop(color, "", u.MaterialProperty.MP_BASE_COLOR)
    if masked:
        prop(samples["Diffuse"], "A", u.MaterialProperty.MP_OPACITY_MASK)
    if "Normal" in samples:
        prop(samples["Normal"], "", u.MaterialProperty.MP_NORMAL)
    for name, chan, mp in (
        ("Rough", "G", u.MaterialProperty.MP_ROUGHNESS),
        ("Metal", "B", u.MaterialProperty.MP_METALLIC),
    ):
        scalar = param(
            u.MaterialExpressionScalarParameter, name, 0.65 if name == "Rough" else 0.0
        )
        output = mul(samples["Packed"], chan, scalar) if "Packed" in samples else scalar
        prop(output, "", mp)
    lib.recompile_material(m)
    changed.append(m)
    masters[key] = m
    return m


def model(gltf):
    d = json.loads(gltf.read_text(encoding="utf8"))
    folder = base + "/" + gltf.parent.name.replace("-", "_")
    images = [gltf.parent / im["uri"] for im in d.get("images", [])]
    fbx = list(sorted(gltf.parent.glob("*.fbx")))
    if not fbx:
        raise RuntimeError(str(gltf))
    expected = [f.stem for f in fbx] + ["MI_" + m["name"] for m in d["materials"]]
    if all(ed.does_asset_exist(folder + "/" + n) for n in expected):
        return
    inputs = load_files(images + fbx, folder)
    slots = {}

    def image_for(info):
        return inputs[images[d["textures"][info["index"]]["source"]].stem]

    for desc in d["materials"]:
        p = desc.get("pbrMetallicRoughness", {})
        maps = {}
        for k, info in (
            ("Diffuse", p.get("baseColorTexture")),
            ("Normal", desc.get("normalTexture")),
            ("Packed", p.get("metallicRoughnessTexture")),
        ):
            if not info:
                continue
            tex = image_for(info)
            if k in ("Normal", "Packed"):
                tex.set_editor_property("srgb", False)
                tex.set_editor_property(
                    "compression_settings",
                    u.TextureCompressionSettings.TC_NORMALMAP
                    if k == "Normal"
                    else u.TextureCompressionSettings.TC_MASKS,
                )
                if k == "Normal":
                    tex.set_editor_property("flip_green_channel", True)
            maps[k] = tex
        masked = desc.get("alphaMode") == "MASK" and "Diffuse" in maps
        parent = master(maps, masked)
        name = desc["name"]
        mi = asset(
            "MI_" + name,
            folder,
            u.MaterialInstanceConstant,
            u.MaterialInstanceConstantFactoryNew(),
        )
        lib.set_material_instance_parent(mi, parent)
        for k, tex in maps.items():
            lib.set_material_instance_texture_parameter_value(mi, k, tex)
        color = p.get("baseColorFactor", [1, 1, 1, 1])
        lib.set_material_instance_vector_parameter_value(
            mi, "Tint", u.LinearColor(*color)
        )
        transform = (
            p.get("baseColorTexture", {})
            .get("extensions", {})
            .get("KHR_texture_transform", {})
        )
        for k, v in (
            ("Metal", p.get("metallicFactor", 1)),
            ("Rough", p.get("roughnessFactor", 1)),
            ("UVScale", transform.get("scale", [1])[0]),
        ):
            lib.set_material_instance_scalar_parameter_value(mi, k, float(v))
        slots[name] = mi
    for f in fbx:
        mesh = inputs[f.stem]
        materials = list(mesh.static_materials)
        for slot in materials:
            slot.set_editor_property(
                "material_interface", slots[str(slot.material_slot_name)]
            )
        mesh.set_editor_property("static_materials", materials)
        mesh.get_editor_property("body_setup").set_editor_property(
            "collision_trace_flag", u.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
        )
    pending = changed + list(inputs.values()) + list(slots.values())
    if not u.EditorLoadingAndSavingUtils.save_packages(
        list({a.get_outermost() for a in pending}), True
    ):
        raise RuntimeError("Saving PBR assets: " + folder)
    changed.clear()
    u.log("HARBOR_PBR " + gltf.parent.name)


for path in sorted(files.glob("*/model.gltf")):
    model(path)
sky = load_files([files / "HarborSky.hdr"], base + "/Sky")["HarborSky"]
u.EditorLoadingAndSavingUtils.save_packages([sky.get_outermost()], True)
u.log("HARBOR_PBR_COMPLETE")
