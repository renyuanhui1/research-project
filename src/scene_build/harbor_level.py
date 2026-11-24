"""Build a new, separate UE5 level inside project_520 after harbor_import.py."""

import json
import math
import random
from collections import Counter
from pathlib import Path

import unreal as u

ROOT = "/Game/CoastalHarbor"
MAP = ROOT + "/Maps/CoastalHarbor"
ED = u.EditorAssetLibrary
LEVEL = u.get_editor_subsystem(u.LevelEditorSubsystem)
ACTORS = u.get_editor_subsystem(u.EditorActorSubsystem)
CITY = "/Game/ModernCityBundle"
PROPS = "/Game/MilitaryAirport/Meshes/Props/SM_"
V = u.Vector
R = u.Rotator
rng = random.Random(80214)
assets = {}
catalog = []
occupied = []
focus = {
    "Building": {1, 8, 13, 20, 36, 49, 71, 78, 85, 95, 98, 112},
    "Vessel": {2, 6, 15, 26, 33, 35, 37, 40},
    "Vehicle": {2, 8, 17, 26, 39, 53, 69, 75, 84, 92, 99, 113},
}
damage = {
    "Building": {1, 13, 36, 71, 85, 95, 98, 112},
    "Vessel": {2, 26, 35, 37},
    "Vehicle": {2, 17, 39, 69, 84, 92, 99},
}
sequence = Counter()


def obj(path):
    if path not in assets:
        assets[path] = u.load_asset(path)
        if not assets[path]:
            raise RuntimeError(path)
    return assets[path]


def loc(x, y, z):
    return V(x * 100, y * 100, z * 100)


def place(source, xyz, folder, label=None, yaw=0, roll=0, pitch=0, scale=(1, 1, 1)):
    sequence[folder] += 1
    if isinstance(source, str):
        source = obj(source)
    pos = loc(*xyz)
    rot = R(roll=roll, pitch=pitch, yaw=yaw)
    if isinstance(source, (u.Class, type)):
        a = ACTORS.spawn_actor_from_class(source, pos, rot)
    else:
        a = ACTORS.spawn_actor_from_object(source, pos, rot)
    if not a:
        raise RuntimeError(str(source))
    a.set_actor_scale3d(V(*scale))
    a.set_actor_label(label or folder.split("/")[-1] + "_%03d" % sequence[folder])
    a.set_folder_path("Harbor/" + folder)
    return a


def surface(a, mat):
    a.static_mesh_component.set_material(0, obj(ROOT + "/Materials/M_" + mat))


def slab(label, x, y, top, w, d, h, mat, folder="Infrastructure"):
    a = place(
        "/Engine/BasicShapes/Cube", (x, y, top - h / 2), folder, label, scale=(w, d, h)
    )
    surface(a, mat)
    return a


def local_mesh(key):
    return ROOT + "/Meshes/" + key


def fitted(source, xyz, folder, sizing, label=None, yaw=0):
    # Use actual component bounds so authored pivots and blueprint foundations sit on the ground.
    a = place(source, xyz, folder, label, yaw=yaw)
    origin, ext = a.get_actor_bounds(False)
    span = max(ext.x, ext.y) * 0.02
    if span < 0.01:
        raise RuntimeError(str(source))
    k = sizing / span
    a.set_actor_scale3d(V(k, k, k))
    origin, ext = a.get_actor_bounds(False)
    pos = a.get_actor_location()
    goal = loc(*xyz)
    a.set_actor_location(
        pos + V(goal.x - origin.x, goal.y - origin.y, goal.z - origin.z + ext.z),
        False,
        False,
    )
    return a


def identify(a, category, i, kind, evidence, state):
    tags = [category, category + "_%03d" % i]
    selected = i in focus[category]
    if selected:
        tags.append("Inspection")
    a.set_editor_property("tags", tags)
    p = a.get_actor_location()
    catalog.append(
        dict(
            id=tags[1],
            category=category,
            kind=kind,
            inspection=selected,
            state=state,
            visible_evidence=evidence,
            xyz_m=[round(v / 100, 3) for v in (p.x, p.y, p.z)],
        )
    )
    if category == "Building":
        center, ext = a.get_actor_bounds(False)
        occupied.append((center.x / 100, center.y / 100, ext.x / 100, ext.y / 100))


def shore(x):
    return -265 - 60 * math.sin(x / 190) - 26 * math.sin(x / 71)


def beach_z(x, y):
    a = shore(x) - 32
    return min(0, -4.5 + 4.5 * (y - a) / (-150 - a))


def groundwork():
    for key, mat in (("Backshore", "Ground"), ("Foreshore", "Mud")):
        path = local_mesh(key)
        box = obj(path).get_bounding_box()
        # Fit imported handedness to the surveyed local map (inland is UE +Y).
        dx, dy = box.max.x - box.min.x, box.max.y - box.min.y
        yaw = 0 if dx > dy else 90
        a = place(path, (0, 0, 0), "Landscape", key, yaw=yaw)
        center, ext = a.get_actor_bounds(False)
        expected_positive = key == "Backshore"
        if (center.y > 0) != expected_positive:
            a.set_actor_scale3d(V(1, -1, 1) if yaw == 0 else V(-1, 1, 1))
        surface(a, mat)
    slab("HarborWater", 0, 0, -2.65, 2000, 2000, 1.2, "Water", "Landscape")
    slab("Quayside", -430, -193, 0.17, 880, 124, 5.4, "Apron")
    for j in range(5):
        x = -800 + j * 162
        slab("Pier_%02d" % j, x, -329, 0.17, 9, 160, 5.4, "Pier")
        for k in range(5):
            for side in (-1, 1):
                a = place(
                    "/Engine/BasicShapes/Cylinder",
                    (x + side * 4.1, -262 - k * 32, 0.30),
                    "Moorings",
                    scale=(0.45, 0.45, 0.75),
                )
                surface(a, "Oxide")
    for x, y, w, d in [
        (-445, -56, 750, 101),
        (460, -45, 780, 83),
        (-520, 160, 660, 61),
    ]:
        slab("ServiceYard_%d" % int(x), x, y, 0.055, w, d, 0.24, "Apron")
    for y in (-120, 85, 290, 495, 700, 905):
        slab("Street_EW_%d" % y, 0, y, 0.090, 1968, 12.5, 0.23, "Road", "Streets")
        slab("Sidewalk_%d" % y, 0, y + 8, 0.13, 1968, 3, 0.25, "Apron", "Streets")
        for x in range(-960, 970, 34):
            if min(abs(x - k) for k in (-900, -300, 300, 900)) > 12:
                slab(
                    "Lane_%d_%d" % (x, y),
                    x,
                    y,
                    0.098,
                    3.6,
                    0.13,
                    0.015,
                    "Stripe",
                    "RoadMarkings",
                )
    for x in (-900, -300, 300, 900):
        slab("Street_NS_%d" % x, x, 400, 0.094, 11, 1065, 0.24, "Road", "Streets")
    for j in range(52):
        x = rng.uniform(60, 980)
        y = (shore(x) - 32) * 0.36 + (-150) * 0.64 + rng.uniform(-6, 6)
        place(
            local_mesh("Wrack"),
            (x, y, beach_z(x, y) + 0.04),
            "TideLine",
            yaw=rng.uniform(-20, 20),
        )


def buildings():
    records = []
    for row in range(7):
        road = (290, 290, 495, 495, 700, 700, 905)[row]
        sy = (-1, 1, -1, 1, -1, 1, -1)[row]
        for col in range(10):
            block = col // 5
            x = -785 + col % 5 * 88 + block * 920 + rng.uniform(-8, 8)
            if abs(x - 300) < 28:
                x += 52
            records.append(
                ("House", x, road + sy * rng.uniform(23, 37), 180 if sy < 0 else 0)
            )
    for row in range(2):
        for col in range(12):
            x = -830 + col * 145
            if min(abs(x - q) for q in (-900, -300, 300, 900)) < 30:
                x += 42
            records.append(("Shop", x, 85 + (-27 if row == 0 else 30), row * 180))
    for j in range(16):
        x = -796 + (j % 8) * 73 + (0 if j < 8 else 875)
        if abs(x + 300) < 27 or abs(x - 300) < 27:
            x += 36
        records.append(("Shed", x, -60 + (j % 2) * 13, 0))
    for j in range(10):
        records.append(("Civic", -775 + j * 167, 187 + (j % 2) * 14, 180))
    for i, (kind, x, y, yaw) in enumerate(records, 1):
        bad = i in damage["Building"]
        lbl = "Building_%03d" % i
        variant = (i // 2) % 5
        if not bad and kind in ("House", "Shop", "Civic") and i % 3:
            number = (
                "01a",
                "02a",
                "03a",
                "04a",
                "05a",
                "06a",
                "07",
                "08",
                "09",
                "10",
                "11",
            )[i % 11]
            stem = "B_Building_S_" + number
            cls = ED.load_blueprint_class(
                CITY + "/Blueprints/BuidlingsStandard/Small/" + stem
            )
            if not cls:
                raise RuntimeError(stem)
            a = fitted(
                cls,
                (x, y, 0.03),
                "Buildings",
                (18 if kind == "House" else 25) + i % 6,
                lbl,
                yaw,
            )
        else:
            key = kind + ("Damaged" if bad else str(variant))
            a = place(
                local_mesh(key),
                (x, y, 0.07),
                "Buildings",
                lbl,
                yaw=yaw + rng.uniform(-4, 4),
            )
        ev = (
            (
                "detached shop awning at facade"
                if kind == "Shop"
                else "lifted and missing roof panels, exposed rafters"
            )
            if bad
            else "intact exterior"
        )
        identify(a, "Building", i, kind, ev, "damaged" if bad else "intact")


def vessels():
    for j in range(40):
        i = j + 1
        kind = (
            "Fishing"
            if j < 20
            else "Launch"
            if j < 26
            else "Workboat"
            if j < 30
            else "Coaster"
            if j < 32
            else ("Fishing", "Workboat", "Launch")[j % 3]
        )
        bad = i in damage["Vessel"]
        if j < 25:
            pier = j // 5
            x = -800 + pier * 162 + (-1 if j % 2 else 1) * 8
            y = -267 - j % 5 * 30
            yaw = 90 + (180 if j % 2 else 0)
            roll, z = 0, -2.65
        elif j < 32:
            x = -860 + (j - 25) * 240
            y = -530 - ((j - 25) % 3) * 135
            yaw, roll, z = rng.uniform(-180, 180), 0, -2.65
        else:
            x = 88 + (j - 32) * 118
            edge = shore(x) - 32
            y = edge + (-150 - edge) * (0.58 + (j % 3) * 0.11)
            roll = (13, -19, 26, 8)[j % 4]
            yaw = rng.uniform(-65, 85)
            z = beach_z(x, y) + 0.45
        key = kind + ("Damaged" if bad else str(i % 2))
        a = place(
            local_mesh(key), (x, y, z), "Vessels", "Vessel_%03d" % i, yaw=yaw, roll=roll
        )
        state = "aground" if j >= 32 else "afloat"
        identify(
            a,
            "Vessel",
            i,
            kind,
            "torn hull plating" if bad else "intact visible hull",
            state,
        )


def vehicles():
    for j in range(120):
        i = j + 1
        kind = (
            "Sedan"
            if j < 65
            else "Van"
            if j < 90
            else "Truck"
            if j < 108
            else "Utility"
            if j < 116
            else "Rescue"
        )
        bad = i in damage["Vehicle"]
        if j < 48:
            row, col = divmod(j, 12)
            x, y = -726 + col * 7.1, -203 + row * 9
            yaw = 90 + rng.uniform(-5, 5)
        else:
            row, col = divmod(j - 48, 12)
            x = -800 + col * 144 + (row % 2) * 12
            if min(abs(x - q) for q in (-900, -300, 300, 900)) < 16:
                x += 30
            y = (-120, 85, 290, 495, 700, 905)[row] - 4.1
            yaw = 180 if col % 3 else 0
        z = 0.18 if j < 48 else 0.10
        roll = 83 if i in (92, 99) else 0
        if roll:
            z += 1.16
        key = kind + ("Damaged" if bad else str(i % 3))
        if kind == "Sedan" and not bad and j % 4 == 0:
            a = fitted(
                "/ProjectAirSim/VehicleAdv/SUV/SUV",
                (x, y, z),
                "Vehicles",
                4.85,
                "Vehicle_%03d" % i,
                yaw,
            )
            kind = "SUV"
            ev = "intact exterior"
        elif kind == "Sedan" and bad:
            path = "/Game/Airport/Meshes/damaged_old_car/SM_damaged_old_car_N_%04d" % (
                j % 6
            )
            a = fitted(path, (x, y, z), "Vehicles", 4.7, "Vehicle_%03d" % i, yaw)
            ev = "crushed vehicle shell"
        else:
            a = place(
                local_mesh(key),
                (x, y, z),
                "Vehicles",
                "Vehicle_%03d" % i,
                yaw=yaw,
                roll=roll,
            )
            ev = "deformed hood and roof" if bad else "intact exterior"
        identify(
            a,
            "Vehicle",
            i,
            kind,
            ev,
            "overturned" if roll else "damaged" if bad else "intact",
        )


def details():
    for j in range(56):
        x = -835 + (j % 14) * 45
        y = -233 + (j // 14) * 6.0
        kind = ("Container", "Crate_Stack_02", "Pallet_3_Stack", "Barrel_Stack_1")[
            j % 4
        ]
        fitted(
            PROPS + kind,
            (x, y, 0.18),
            "DockFreight",
            (9.1, 2.5, 2.1, 2.4)[j % 4],
            yaw=2 * (j % 5),
        )
    props = CITY + "/Meshes/StreetElements/SM_"
    for y in (-120, 85, 290, 495, 700, 905):
        for x in range(-820, 945, 165):
            fitted(props + "Street_Light", (x, y + 9.7, 0.15), "Lights", 3.4, yaw=90)
    for j in range(8):
        fitted(
            props + "BusShelter",
            (-710 + j % 4 * 430, 85 + (j // 4) * 410 + 10, 0.15),
            "StreetFurniture",
            5.8,
        )
    # Keep trees off buildings, roads, and the unobstructed yard at the world origin.
    for j in range(700):
        x, y = rng.uniform(-978, 978), rng.uniform(125, 975)
        if (
            min(abs(x - q) for q in (-900, -300, 300, 900)) < 12
            or min(abs(y - q) for q in (290, 495, 700, 905)) < 16
        ):
            continue
        if any(
            abs(x - bx) < ex + 6 and abs(y - by) < ey + 6 for bx, by, ex, ey in occupied
        ):
            continue
        if j % 3 and abs(math.sin(x / 155) * math.cos(y / 235)) < 0.40:
            continue
        fitted(
            props + "Tree_1",
            (x, y, 0),
            "Vegetation",
            rng.uniform(5, 11),
            yaw=rng.uniform(0, 360),
        )
    for i in (8, 14, 20, 28, 31, 42, 46):
        r = catalog[i - 1]
        x, y, _ = r["xyz_m"]
        # Downed timber nearby and roof fragments local to affected buildings.
        place(local_mesh("Wrack"), (x + 16, y - 8, 0.07), "StormDebris", yaw=17)
    for r in catalog:
        if r["category"] == "Building" and r["state"] == "damaged":
            x, y, _ = r["xyz_m"]
            a = slab(
                "RoofFragment_" + r["id"],
                x + 8,
                y - 11,
                0.45,
                1.8,
                3.1,
                0.08,
                "Oxide",
                "StormDebris",
            )
            a.set_actor_rotation(R(pitch=8, roll=9, yaw=27), False)


def environment():
    game = u.load_class(None, "/Script/ProjectAirSim.ProjectAirSimGameMode")
    if not game:
        raise RuntimeError("ProjectAirSimGameMode")
    world = u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
    world.get_world_settings().set_editor_property("default_game_mode", game)
    place(u.PlayerStart, (0, 0, 1.1), "Environment", "HarborPlayerStart")
    sun = place(
        u.DirectionalLight, (0, 0, 200), "Environment", "CoastalSun", yaw=-53, pitch=-38
    )
    sun.light_component.set_mobility(u.ComponentMobility.MOVABLE)
    sun.light_component.set_editor_property("intensity", 4.4)
    sun.light_component.set_editor_property("atmosphere_sun_light", True)
    place(u.SkyAtmosphere, (0, 0, -5), "Environment", "MarineAtmosphere")
    sky = place(u.SkyLight, (0, 0, 60), "Environment", "MarineSkyLight")
    sky.light_component.set_mobility(u.ComponentMobility.MOVABLE)
    sky.light_component.set_editor_property("real_time_capture", True)
    fog = place(u.ExponentialHeightFog, (0, 0, -10), "Environment", "SeaHaze")
    fog.component.set_editor_property("fog_density", 0.006)


def main():
    if ED.does_asset_exist(MAP):
        # A failed run may have saved only the empty level. Never replace an edited map.
        if not LEVEL.load_level(MAP):
            raise RuntimeError(MAP)
        if any(
            str(a.get_folder_path()).startswith("Harbor")
            for a in ACTORS.get_all_level_actors()
        ):
            raise RuntimeError("The populated level already exists")
    elif not LEVEL.new_level(MAP):
        raise RuntimeError("new_level failed")
    environment()
    groundwork()
    buildings()
    vessels()
    vehicles()
    details()
    if Counter(r["category"] for r in catalog) != {
        "Building": 120,
        "Vessel": 40,
        "Vehicle": 120,
    }:
        raise RuntimeError("target count")
    if sum(r["inspection"] for r in catalog) != 32:
        raise RuntimeError("inspection count")
    u.EditorLevelLibrary.set_level_viewport_camera_info(
        loc(-540, -510, 145), R(yaw=43, pitch=-23)
    )
    if not LEVEL.save_current_level():
        raise RuntimeError("save_current_level")
    out = Path(__file__).parent
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf8")
    u.log("HARBOR_LEVEL_SAVED " + str(dict(sequence)))


if __name__ == "__main__":
    main()
