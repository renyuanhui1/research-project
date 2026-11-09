"""Load a scene's assets and preparation parameters; no simulator side effects."""
import argparse
import copy
import json
import os
from pathlib import Path
from common.paths import CONFIGS, project_path

def load_profile(value=None):
    value = value or os.environ.get("UAV_SCENE_PROFILE", "airport")
    path = Path(value).expanduser()
    if path.suffix != ".json" and len(path.parts) == 1:
        path = CONFIGS / "scenes" / (str(value) + ".json")
    else:
        path = project_path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for section in ("name", "simulator", "assets", "map", "target", "preparation", "controller", "mission"):
        if section not in data:
            raise ValueError(f"Missing profile field {section}: {path}")
    data["profile_path"] = str(path.resolve())
    return data

def selected_profile(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", default=None)
    args, _ = parser.parse_known_args(argv)
    return load_profile(args.profile)

def load_mission(profile, value=None):
    path = project_path(value or profile["mission"])
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("profile") not in (None, profile["name"]):
        raise ValueError(f"Mission profile does not match {profile['name']}: {path}")
    return data

def scene_bundle(profile):
    import commentjson
    path = project_path(profile["simulator"]["scene"])
    with path.open(encoding="utf-8") as stream:
        scene = commentjson.load(stream)
    for actor in scene["actors"]:
        robot = actor.get("robot-config")
        if isinstance(robot, str):
            robot_path = (project_path(profile["simulator"]["robot"])
                          if actor.get("name") == profile["simulator"]["drone"] and profile["simulator"].get("robot")
                          else path.parent / robot)
            with robot_path.open(encoding="utf-8") as stream:
                actor["robot-config"] = commentjson.load(stream)
    return copy.deepcopy(scene)
