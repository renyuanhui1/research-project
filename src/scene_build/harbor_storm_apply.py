"""Apply the documented static wind aftermath to CoastalHarbor in a rendering UE editor."""
import json
import math
from pathlib import Path

import unreal as u

ROOT="/Game/CoastalHarbor"
DEST=ROOT+"/StormAftermath"
SOURCES={"VillageRoofBreach":("village_house","Building_001"),"TugBentMast":("tugboat","Vessel_035")}
FOLDER="Harbor/StormAftermath"
TARGET_MESH = {name: "VillageRoofBreach" for name in ("Building_001", "Building_013", "Building_036")}
TARGET_MESH["Vessel_035"] = "TugBentMast"
EVIDENCE = {
    "VillageRoofBreach": "roof covering missing across adjacent bays with exposed rafters and battens; one roof corner lifted",
    "TugBentMast": "upper mast folded sideways above the wheelhouse; locally indented hull side and bent rail",
}


def sync_catalog(file):
    rows = json.loads(file.read_text(encoding="utf8"))
    for row in rows:
        variant = TARGET_MESH.get(row["id"])
        if variant:
            row.update(model=variant, visible_evidence=EVIDENCE[variant])
    file.write_text(json.dumps(rows, indent=2), encoding="utf8")


def source_dir():
    project=Path(u.Paths.convert_relative_path_to_full(u.Paths.project_dir()))
    return project/"SourceArt/CoastalHarbor/StormAftermath"


def import_meshes(source):
    lib=u.EditorAssetLibrary
    tools=u.AssetToolsHelpers.get_asset_tools()
    for f in sorted(source.glob("*.fbx")):
        if lib.does_asset_exist(DEST+"/"+f.stem):
            continue
        task=u.AssetImportTask()
        opt=u.FbxImportUI()
        opt.automated_import_should_detect_type=False
        opt.mesh_type_to_import=u.FBXImportType.FBXIT_STATIC_MESH
        opt.import_materials=opt.import_textures=opt.import_as_skeletal=False
        opt.static_mesh_import_data.combine_meshes=True
        opt.static_mesh_import_data.auto_generate_collision=False
        opt.static_mesh_import_data.generate_lightmap_u_vs=False
        opt.static_mesh_import_data.vertex_color_import_option=u.VertexColorImportOption.REPLACE
        task.filename=str(f);task.destination_path=DEST;task.automated=True;task.save=False;task.options=opt
        tools.import_asset_tasks([task])
        assert task.imported_object_paths,f.stem
        mesh=u.load_asset(DEST+"/"+f.stem)
        source_materials={}
        if f.stem in SOURCES:
            base=SOURCES[f.stem][0]
            original=u.load_asset(f"{ROOT}/Library/{base}/{base}_Storm")
            source_materials={str(s.material_slot_name):s.material_interface for s in original.static_materials}
        for i,s in enumerate(mesh.static_materials):
            key=str(s.material_slot_name)
            mat=source_materials.get(key) or u.load_asset(ROOT+"/Materials/M_"+key)
            if not mat:raise RuntimeError(f"Missing {f.stem} material {key}")
            mesh.set_material(i,mat)
        mesh.get_editor_property("body_setup").set_editor_property("collision_trace_flag",u.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
        assert lib.save_loaded_asset(mesh,False)


def apply():
    source=source_dir()
    plan=json.loads((source/"wind-layout.json").read_text())
    ed=u.get_editor_subsystem(u.UnrealEditorSubsystem)
    actors=u.get_editor_subsystem(u.EditorActorSubsystem)
    level=u.get_editor_subsystem(u.LevelEditorSubsystem)
    world=ed.get_editor_world()
    assert world.get_path_name().split(".")[0]==ROOT+"/Maps/CoastalHarbor"
    import_meshes(source)
    all_actors=actors.get_all_level_actors()
    lookup={a.get_actor_label():a for a in all_actors}
    # Reapplying after rebuilding the town must recreate the same field evidence.
    for a in all_actors:
        if str(a.get_folder_path())==FOLDER:actors.destroy_actor(a)
    for r in plan["roofs"]:
        a=lookup[r["label"]]
        a.modify();comp=a.static_mesh_component;comp.modify()
        comp.set_static_mesh(u.load_asset(DEST+"/"+r["mesh"]))
    for name, variant in TARGET_MESH.items():
        a=lookup[name];a.modify();a.static_mesh_component.modify()
        a.static_mesh_component.set_static_mesh(u.load_asset(DEST+"/"+variant))

    surface_names={"land":{"Backshore","TownStreets","TownLots","FieldParcels"},
                   "quay":{"Quayside"},"mud":{"Foreshore","Backshore"}}
    def ground(x,y,allowed):
        hit=u.SystemLibrary.line_trace_single(world,u.Vector(x*100,y*100,23000),
            u.Vector(x*100,y*100,-600),u.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],u.DrawDebugTrace.NONE,True)
        if not hit:return None
        parts=hit.to_tuple()
        if not any(lookup[label] in parts for label in allowed):return None
        return parts[5].z/100
    placed=[]
    for i,r in enumerate(plan["props"]):
        angle=math.radians(r["yaw"]);dx,dy=math.cos(angle),math.sin(angle)
        allow=surface_names[r["surface"]]
        mesh=u.load_asset(DEST+"/"+r["mesh"])
        for off_x,off_y in ((0,0),(2,2),(-2,2),(0,-2),(3.5,1),(-3,1),(1,4)):
            x,y=r["xy"][0]+off_x,r["xy"][1]+off_y
            z=ground(x,y,allow)
            if z is None or (r["surface"]=="mud" and z < -2.65):continue
            length=r["radius"]*.78
            tests=[(dx*length,dy*length),(-dx*length,-dy*length),(-dy,dx),(dy,-dx)]
            heights=[ground(x+a,y+b,allow) for a,b in tests]
            if None in heights:continue
            if abs((heights[0]+heights[1])/2-z)>.12 or abs((heights[2]+heights[3])/2-z)>.10:continue
            slope=(heights[0]-heights[1])/(2*length);cross=(heights[2]-heights[3])/2
            rot=u.Rotator(pitch=math.degrees(math.atan(slope)),yaw=r["yaw"],roll=-math.degrees(math.atan(cross)))
            if max(abs(slope),abs(cross))>.32:continue
            a=actors.spawn_actor_from_object(mesh,u.Vector(x*100,y*100,z*100),rot)
            name=f"Wind_{i+1:03}_{r['mesh']}"
            a.set_actor_label(name);a.set_folder_path(FOLDER)
            placed.append(dict(label=name,mesh=r["mesh"],source=r["source"],xyz_m=[x,y,z]))
            break
    assert len(placed)>=len(plan["props"])*.74, len(placed)
    assert level.save_current_level()
    (source/"placed.json").write_text(json.dumps(placed,indent=2))
    sync_catalog(source.parent / "PortTown" / "catalog.json")
    u.log("WIND_AFTER_SAVED "+str(len(placed)))
