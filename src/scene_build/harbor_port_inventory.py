"""Export the preceding street map as metric source parcels for harbor_port_plan.py."""

import json, unreal as u
from pathlib import Path

actors = u.get_editor_subsystem(u.EditorActorSubsystem)
level = u.get_editor_subsystem(u.LevelEditorSubsystem)
assert level.load_level("/Game/CoastalHarbor/Maps/CoastalHarbor")


def xyz(v):
    return [round(a / 100, 4) for a in (v.x, v.y, v.z)]


rows = []
for a in actors.get_all_level_actors():
    c, e = a.get_actor_bounds(False)
    m = getattr(a, "static_mesh_component", None)
    m = m.static_mesh if m else None
    r = a.get_actor_rotation()
    rows.append(
        dict(
            key=a.get_name(),
            label=a.get_actor_label(),
            folder=str(a.get_folder_path()).split("/")[-1],
            mesh=m.get_path_name().split(".")[-1] if m else "",
            p=xyz(a.get_actor_location()),
            c=xyz(c),
            e=xyz(e),
            yaw=r.yaw,
        )
    )
Path(__file__).with_name("inventory.json").write_text(json.dumps(rows))
