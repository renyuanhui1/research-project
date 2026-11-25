"""Import ShedTornRoof_Passable.fbx into CoastalHarbor and assign it to Building_098 only.

Leaves Building_095 on unmodified ShedTornRoof (with intact steel purlins across bay j=4)
as the obstructed control target for Task 2 indoor-entry passability evaluation.

Run headless with UE 5.2 (close UE Editor first to avoid map lock):
  UnrealEditor-Cmd.exe <project_520.uproject> -run=pythonscript -script=<this_file>
"""

from pathlib import Path
import unreal as u

ROOT = "/Game/CoastalHarbor"
MAP_PATH = ROOT + "/Maps/CoastalHarbor"
MESH_FOLDER = ROOT + "/Meshes"
MESH_NAME = "ShedTornRoof_Passable"
TARGET_ACTOR_LABEL = "Building_098"


def main() -> None:
    ed = u.EditorAssetLibrary
    tools = u.AssetToolsHelpers.get_asset_tools()
    level = u.get_editor_subsystem(u.LevelEditorSubsystem)
    actors = u.get_editor_subsystem(u.EditorActorSubsystem)

    project_dir = Path(u.Paths.convert_relative_path_to_full(u.Paths.project_dir()))
    fbx_path = project_dir / "SourceArt" / "CoastalHarbor" / "terrain" / f"{MESH_NAME}.fbx"
    if not fbx_path.is_file():
        raise FileNotFoundError(f"Missing FBX: {fbx_path}")

    assert level.load_level(MAP_PATH), f"Failed to load {MAP_PATH}"

    asset_path = f"{MESH_FOLDER}/{MESH_NAME}"
    task = u.AssetImportTask()
    opt = u.FbxImportUI()
    opt.set_editor_property("automated_import_should_detect_type", False)
    opt.set_editor_property("mesh_type_to_import", u.FBXImportType.FBXIT_STATIC_MESH)
    opt.set_editor_property("import_materials", False)
    opt.set_editor_property("import_textures", False)
    opt.set_editor_property("import_as_skeletal", False)
    opt.static_mesh_import_data.set_editor_property("combine_meshes", True)
    opt.static_mesh_import_data.set_editor_property("auto_generate_collision", False)
    opt.static_mesh_import_data.set_editor_property(
        "vertex_color_import_option", u.VertexColorImportOption.REPLACE
    )
    task.set_editor_property("filename", str(fbx_path))
    task.set_editor_property("destination_path", MESH_FOLDER)
    task.set_editor_property("options", opt)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", False)
    tools.import_asset_tasks([task])

    mesh = ed.load_asset(asset_path)
    if mesh is None:
        raise RuntimeError(f"Failed to import {asset_path}")

    slots = list(mesh.static_materials)
    for slot in slots:
        mat_name = f"{ROOT}/Materials/M_{slot.material_slot_name}"
        mat = ed.load_asset(mat_name)
        if mat is None:
            raise RuntimeError(f"Missing material {mat_name}")
        slot.set_editor_property("material_interface", mat)
    mesh.set_editor_property("static_materials", slots)
    mesh.get_editor_property("body_setup").set_editor_property(
        "collision_trace_flag", u.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE
    )
    assert ed.save_loaded_asset(mesh, False), f"Failed to save {asset_path}"

    target_actor = None
    for a in actors.get_all_level_actors():
        if a.get_actor_label() == TARGET_ACTOR_LABEL:
            target_actor = a
            break
    if target_actor is None:
        raise RuntimeError(f"Actor {TARGET_ACTOR_LABEL} not found in {MAP_PATH}")

    target_actor.modify()
    comp = target_actor.static_mesh_component
    comp.modify()
    comp.set_static_mesh(mesh)
    assert level.save_current_level(), f"Failed to save {MAP_PATH}"
    u.log(f"PASSABLE_SHED_APPLIED actor={TARGET_ACTOR_LABEL} mesh={asset_path}")


if __name__ == "__main__":
    main()
