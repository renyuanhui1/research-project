"""Design the synthetic harbor in survey metres. Run with the project's airsim Python."""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

HALF_EXTENT_M = 1400  # Source terrain covers the full clipping polygon.
MODEL_BOUNDARY = Polygon(json.loads(Path(__file__).with_name("harbor_outline.json").read_text()))

# Named junctions join the working quay, the market, the old town and the inland hamlets.
NODES = {
    "jetty": (-825, -118),
    "crane": (-526, -119),
    "slip": (-180, -118),
    "endquay": (-38, -100),
    "west": (-886, -37),
    "fish": (-692, -29),
    "strand": (-476, 19),
    "square": (-272, 65),
    "bridge": (-34, 42),
    "bend": (175, -26),
    "green": (437, -64),
    "east": (710, -91),
    "kiln": (-798, 126),
    "chapel": (-560, 177),
    "inn": (-345, 215),
    "mill": (-119, 187),
    "school": (68, 248),
    "orchard": (258, 170),
    "well": (550, 103),
    "fold": (-700, 345),
    "crest": (-492, 418),
    "cross": (-233, 381),
    "glen": (0, 485),
    "heath": (302, 385),
    "croft": (498, 526),
    "barn": (-350, 684),
    "hill": (-60, 744),
}
ROUTES = [
    ("Quayside", "dock", ["jetty", "crane", "slip", "endquay"]),
    ("West slip", "dock", ["jetty", (-866, -104), "west"]),
    ("Auction lane", "dock", ["crane", (-560, -79), "strand"]),
    ("Iceworks lane", "dock", ["slip", (-222, -65), "square"]),
    (
        "Harbour approach",
        "main",
        [
            (-5320, 830),
            (-4350, 492),
            (-3310, 551),
            (-2560, 265),
            (-1890, 300),
            (-1450, 185),
            (-1110, 33),
            "west",
            "fish",
            "strand",
            "square",
            "bridge",
            "bend",
            "green",
            "east",
            (910, -21),
            (1050, 34),
            (1230, 161),
            (1450, 288),
            (1770, 308),
            (2080, 262),
            (2420, 205),
            (3000, 110),
            (3800, 135),
            (4950, 65),
            (5720, 180),
        ],
    ),
    (
        "River road",
        "main",
        ["endquay", (-87, -49), "bridge", (-84, 120), "mill", (-87, 312), "cross"],
    ),
    (
        "Chapel street",
        "old",
        ["west", (-897, 54), "kiln", (-709, 142), "chapel", "inn", "mill"],
    ),
    ("Fish steps", "old", ["fish", (-671, 50), (-739, 90), "kiln"]),
    ("Net loft lane", "old", ["strand", (-463, 74), (-533, 111), "chapel"]),
    ("Church lane", "old", ["square", (-308, 116), (-301, 163), "inn"]),
    (
        "Coombe road",
        "village",
        ["mill", (-45, 193), "school", (152, 231), "orchard", (355, 116), "green"],
    ),
    ("Salt road", "village", ["east", (686, 7), "well", (418, 148), "orchard"]),
    (
        "West ridge road",
        "village",
        ["kiln", (-824, 217), "fold", (-586, 364), "crest", "cross"],
    ),
    ("Orchard rise", "village", ["chapel", (-533, 247), (-601, 300), "fold"]),
    ("Chapel court", "court", ["inn", (-379, 251), (-412, 266), (-427, 295)]),
    ("Back lane", "old", ["inn", (-288, 287), "cross"]),
    (
        "Valley road",
        "country",
        ["cross", (-141, 438), "glen", (147, 467), "heath", "orchard"],
    ),
    ("School close", "court", ["school", (94, 302), (58, 326), (27, 315)]),
    (
        "East coombe",
        "country",
        [
            "heath",
            (426, 373),
            (490, 445),
            "croft",
            (697, 654),
            (936, 748),
            (1320, 781),
            (1630, 663),
            (1920, 485),
            (2080, 262),
        ],
    ),
    (
        "Upland road",
        "country",
        [
            "crest",
            (-476, 531),
            "barn",
            (-208, 669),
            "hill",
            (104, 802),
            (203, 1004),
            (478, 1280),
            (705, 1650),
            (623, 2180),
            (1080, 2730),
            (1380, 3300),
        ],
    ),
    ("Mill track", "country", ["glen", (-21, 580), (-102, 645), "hill"]),
    ("Farm court", "court", ["barn", (-422, 718), (-448, 771)]),
    ("Netmakers court", "court", ["fish", (-721, -57), (-761, -63)]),
    ("Quarry lane", "court", ["well", (607, 144), (664, 167)]),
    (
        "Westcombe lane", "village",
        ["fold", (-804, 492), (-812, 639), (-706, 740), (-635, 778), (-538, 723), "barn"],
    ),
    (
        "Ridge terrace", "village",
        [(-635, 778), (-579, 837), (-411, 895), (-289, 860), (-171, 814), "hill"],
    ),
    ("Manor close", "court", [(-411, 895), (-443, 926), (-519, 936)]),
    ("Mill green", "old", ["glen", (88, 558), (143, 639), (81, 688), "hill"]),
    (
        "High cross", "village",
        ["croft", (501, 655), (455, 741), (375, 815), (232, 855), (104, 802)],
    ),
    (
        "Orchard lane", "village",
        [(455, 741), (542, 809), (636, 855), (761, 819), (804, 753), (697, 654)],
    ),
    ("Orchard yard", "court", [(636, 855), (650, 902), (710, 928)]),
]
WIDTH = {
    "main": 9.2,
    "dock": 9.0,
    "old": 5.7,
    "village": 6.7,
    "court": 4.1,
    "country": 5.6,
}
WALK = {
    "main": 1.9,
    "dock": 0.6,
    "old": 1.3,
    "village": 1.0,
    "court": 0.55,
    "country": 0.35,
}


def smoothstep(x):
    t = max(0, min(1, x))
    return t * t * (3 - 2 * t)


def bank(x):
    t = smoothstep((abs(x) - 970) / 680)
    return -150 + t * (185 * math.sin(x / 550) + 270 * math.sin(x / 1300))


def relief(x, y):
    inland = max(0, y - bank(x))
    foothill = smoothstep((inland - 156) / 850)
    return foothill * (
        22 + 13 * math.sin(x / 275 + y / 490) + 8 * math.cos(y / 180 - x / 490)
    ) + smoothstep((inland - 800) / 1600) * (67 + 42 * math.sin(x / 600 - y / 940))


def shore(x):
    old = -265 - 60 * math.sin(x / 190) - 26 * math.sin(x / 71) - 32
    t = smoothstep((abs(x) - 985) / 550)
    basin = smoothstep((x + 960) / 155) * (1 - smoothstep((x + 225) / 175))
    # Lower intertidal shelf under the working quay; keep the eastern stranded hulls on their measured bed.
    return (1 - t) * (old + basin * max(0, -235 - old)) + t * (
        bank(x) - 85 - 24 * math.sin(x / 222)
    )


def xy(p):
    return np.array([p.x, p.y])


def routes():
    roads = []
    for name, kind, points in ROUTES:
        p = np.array([NODES.get(v, v) for v in points], dtype=float)
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        deriv = np.gradient(p, dist, axis=0)
        spline = CubicHermiteSpline(dist, p, deriv)
        samples = np.unique(np.r_[dist, np.arange(0, dist[-1], 4.6)])
        centres = spline(samples)
        if name == "Harbour approach":
            for pxy in centres:
                if abs(pxy[0]) > 990:
                    pxy[1] = max(pxy[1], bank(pxy[0]) + 32)
        line = LineString(centres)
        roads.append((name, kind, line))
    return roads


class Geometry:
    def __init__(self):
        self.meshes = defaultdict(lambda: defaultdict(list))

    def tri(self, asset, mat, a, b, c):
        self.meshes[asset][mat].append([a, b, c])

    def footprint(self, asset, mat, shape, height, step=42):
        if not MODEL_BOUNDARY.covers(shape):
            shape = shape.intersection(MODEL_BOUNDARY)
        # Cut ribbons into small patches so their surfaces follow the same relief as the plots.
        if shape.is_empty:
            return
        x0, y0, x1, y1 = shape.bounds
        for x in range(math.floor(x0 / step) * step, math.ceil(x1 / step) * step, step):
            for y in range(
                math.floor(y0 / step) * step, math.ceil(y1 / step) * step, step
            ):
                cut = shapely.clip_by_rect(shape, x, y, x + step, y + step)
                if cut.is_empty:
                    continue
                for t in shapely.get_parts(shapely.constrained_delaunay_triangles(cut)):
                    pts = [(a, b, height(a, b)) for a, b in list(t.exterior.coords)[:3]]
                    self.tri(asset, mat, *pts)


def terrain(g):
    xgrid = sorted(
        {-HALF_EXTENT_M, HALF_EXTENT_M}
        | set(range(-1400, -1000, 25))
        | set(range(-1000, 1001, 10))
        | set(range(1000, 1401, 25))
    )
    # The inner strips meet the outer coast at identical vertices. The study boundary has no visible rim.
    for x0, x1 in zip(xgrid, xgrid[1:]):
        inside = -1000 <= x0 < x1 <= 1000

        def strip(x):
            edge, upper = shore(x), bank(x)
            mud = [
                (x, edge + (upper - edge) * t, -4.5 + 4.5 * t)
                for t in (0, 0.25, 0.5, 0.75, 1)
            ]
            offsets = list(range(0, 1151, 10)) + list(range(1175, 1776, 25))
            land = []
            for s in offsets:
                y = min(upper + s, HALF_EXTENT_M)
                land.append((x, y, relief(x, y)))
            land.append((x, HALF_EXTENT_M, relief(x, HALF_EXTENT_M)))
            return mud, land

        left, right = strip(x0), strip(x1)
        for index, mat in enumerate(("Mud", "MixedCoastalGrass")):
            A, B = left[index], right[index]
            for j in range(len(A) - 1):
                core = inside and (index == 0 or j < 115)
                asset = (
                    ("TidalShelf" if index == 0 else "TownGround")
                    if core
                    else "CoastBeyondTown"
                )
                if B[j] != B[j + 1]:
                    g.tri(asset, mat, A[j], B[j], B[j + 1])
                if A[j] != A[j + 1]:
                    g.tri(asset, mat, A[j], B[j + 1], A[j + 1])


def frame(line, d, side):
    p = xy(line.interpolate(d))
    t = xy(line.interpolate(min(line.length, d + 1))) - xy(
        line.interpolate(max(0, d - 1))
    )
    t = t / np.linalg.norm(t)
    return p, t, np.array([-t[1], t[0]]) * side


def rectangle(c, t, n, w, d):
    return Polygon(
        [
            c + t * a * w / 2 + n * b * d / 2
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )


def working_quay(g, src):
    outline = [
        (-875, -134),
        (-865, -181),
        (-844, -235),
        (-819, -255),
        (-782, -255),
        (-759, -246),
        (-691, -246),
        (-671, -238),
        (-655, -255),
        (-624, -255),
        (-607, -237),
        (-536, -240),
        (-511, -253),
        (-458, -255),
        (-441, -242),
        (-386, -244),
        (-342, -238),
        (-328, -255),
        (-295, -255),
        (-278, -239),
        (-198, -241),
        (-171, -255),
        (-133, -255),
        (-116, -235),
        (-78, -218),
        (-41, -215),
        (-13, -189),
        (22, -175),
        (31, -148),
        (-34, -134),
    ]
    quay = Polygon(outline)
    old = box(-870, -255, 10, -131)
    clutter = []
    for r in src:
        x, y = r["c"][:2]
        ex, ey = r["e"][:2]
        if old.contains(Point(x, y)) and r["folder"] in (
            "Vehicles",
            "Parking",
            "Moorings",
            "DockEquipment",
            "DockFreight",
            "LandingGear",
        ):
            f = box(x - ex - 0.25, y - ey - 0.25, x + ex + 0.25, y + ey + 0.25)
            clutter.append(f)
            if not quay.covers(f.intersection(old)):
                quay = quay.union(
                    f.intersection(old)
                    .union(Point(x, min(-145, y + 22)).buffer(1))
                    .convex_hull
                )
    for x in (-814, -527, -194):
        quay = quay.union(box(x - 6, -139, x + 6, -119.9))
    g.footprint("WorkingQuay", "Apron", quay, lambda x, y: 0.17)
    for a, b in zip(quay.exterior.coords, list(quay.exterior.coords)[1:]):
        g.tri("WorkingQuay", "Apron", (*a, 0.17), (*a, -4.9), (*b, -4.9))
        g.tri("WorkingQuay", "Apron", (*a, 0.17), (*b, -4.9), (*b, 0.17))
    # Two working slips descend below the estuary surface at their outer ends.
    for x, y in [(-696, -243), (-70, -212)]:
        a, b, c, d = (
            (x - 3.2, y, 0.18),
            (x + 3.2, y, 0.18),
            (x + 3.2, y - 22, -3.15),
            (x - 3.2, y - 22, -3.15),
        )
        g.tri("WorkingQuay", "Apron", a, b, c)
        g.tri("WorkingQuay", "Apron", a, c, d)
    props = []
    rr = random.Random(670)
    for x in (
        -828,
        -787,
        -747,
        -619,
        -580,
        -540,
        -466,
        -390,
        -344,
        -281,
        -197,
        -128,
        -83,
    ):
        y = rr.uniform(-157, -146)
        length = rr.uniform(14, 23)
        space = box(
            x - length / 2 - 2,
            y - length * 0.42 - 2,
            x + length / 2 + 2,
            y + length * 0.42 + 2,
        )
        if (
            any(space.intersects(p) for p in clutter)
            or space.contains(Point(-505, -155))
            or not quay.covers(space)
        ):
            continue
        clutter.append(space)
        props.append(
            [
                "warehouse" if len(props) % 3 else "boathouse",
                x,
                y,
                0.17,
                length,
                rr.uniform(-9, 9),
            ]
        )
    for cx, cy in [
        (-785, -195),
        (-592, -204),
        (-358, -185),
        (-248, -216),
        (-145, -196),
        (-65, -174),
    ]:
        for j in range(rr.randrange(3, 8)):
            x = cx + rr.uniform(-7, 7)
            y = cy + rr.uniform(-6, 6)
            footprint = Point(x, y).buffer(1.8)
            if not quay.covers(footprint) or any(
                footprint.intersects(p) for p in clutter
            ):
                continue
            clutter.append(footprint)
            props.append(
                [
                    rr.choice(("Crate_Stack_02", "Pallet_3_Stack", "Barrel_Stack_1")),
                    x,
                    y,
                    0.18,
                    rr.uniform(1.6, 2.8),
                    rr.uniform(-25, 25),
                ]
            )
    return [r for r in props if math.hypot(r[1] + 390, r[2] + 154) > r[4] / 2 + 7]


def trim_layout(data):
    """Keep the retained trees' original species and trim the exported road axes."""
    species_rng = random.Random(515)
    existing = iter(data["tree_species"]) if "tree_species" in data else None
    kept, species = [], []
    for item in data["decor"]:
        if item[0] == "tree":
            if existing is None:
                tree = "lombardy_poplar" if species_rng.random() < 0.17 else "silver_birch"
            else:
                tree = next(existing)
        if MODEL_BOUNDARY.covers(Point(item[1], item[2])):
            kept.append(item)
            if item[0] == "tree":
                species.append(tree)
    data["decor"] = kept
    data["tree_species"] = species
    data["surface_half_extent_m"] = HALF_EXTENT_M
    data["outline_m"] = list(MODEL_BOUNDARY.exterior.coords)
    for road in data["roads"]:
        line = LineString(road["xy"]).intersection(MODEL_BOUNDARY)
        if line.geom_type != "LineString" or line.is_empty:
            raise ValueError(f"Expected a continuous in-bounds road: {road['name']}")
        road["xy"] = list(line.coords)


def foundation(g, site, t, n, hw, hd, ground, build, c, kind, k):
    # A plinth touches the ground along all four walls on sloping plots.
    ring = list(build.exterior.coords)[:-1]
    for a, b in zip(ring, ring[1:] + ring[:1]):
        ab = (*a, ground)
        bb = (*b, ground)
        al = (*a, relief(*a) - 0.06)
        bl = (*b, relief(*b) - 0.06)
        g.tri("TownLots", "Apron", ab, al, bl)
        g.tri("TownLots", "Apron", ab, bl, bb)
    g.footprint("TownLots", "Apron", build, lambda x, y: ground)
    front = site - n * hd
    kerb = c + n * (WIDTH[kind] / 2 + WALK[kind] - 0.08)
    apron = rectangle(
        (front + kerb) / 2,
        t,
        n,
        2 * hw + 0.2 if k in ("shop", "works", "civic") else min(hw * 1.3, 3.6),
        np.linalg.norm(front - kerb) + 0.2,
    )
    g.footprint(
        "TownLots",
        "CompactedQuayGravel" if k == "works" else "Apron",
        apron,
        lambda x, y: relief(x, y) + 0.075,
    )


def arrange_residential_rows(plan, geometry, street):
    """Reserve continuous street frontages around five neighbourhood centres."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    houses = [lot for lot in plan["lots"] if lot["kind"] == "home"]
    max_width = max(h["hw"] for h in houses)
    max_depth = max(h["hd"] for h in houses)
    fixed = [street]
    for lot in plan["lots"]:
        if lot["kind"] != "home":
            c, t, n = (np.array(lot[k]) for k in ("c", "t", "n"))
            fixed.append(rectangle(c, t, n, 2*lot["hw"] + 3, 2*lot["hd"] + 3))
    for key, x, y, _, size, _ in plan["decor"]:
        fixed.append(Point(x,y).buffer(size/2 + 1 if key == "tree" else 3.5))
    for triangles in geometry.meshes["FieldParcels"].values():
        fixed.extend(Polygon([p[:2] for p in tri]) for tri in triangles)
    obstacles = STRtree(fixed)
    # West village, upper ridge, market hinterland, east village and orchard.
    hubs = np.array([(-670,245),(-440,780),(-100,290),(400,375),(320,820)])
    roads = {r["name"]: (r["kind"], LineString(r["xy"])) for r in plan["roads"]}
    candidates = []
    for name,(kind,line) in roads.items():
        if kind == "dock":
            continue
        curb = WIDTH[kind]/2 + WALK[kind]
        for side in (-1,1):
            run = []
            for station in np.arange(14, line.length-12, 24):
                c,t,n = frame(line,station,side)
                center = c + n*(curb+5+max_depth)
                reserve = rectangle(center,t,n,2*max_width+2,2*max_depth+2)
                hub = int(np.argmin(np.linalg.norm(hubs-center,axis=1)))
                valid = (center[1]>100 and max(abs(center))<975 and
                         not obstacles.query(reserve,predicate="intersects").size)
                if not valid or (run and (run[-1][0]!=hub or np.dot(run[-1][4],t)<0.94)):
                    run = []
                if not valid:
                    continue
                run.append((hub,name,kind,c,t,n,side,reserve,station))
                for size in range(3,min(8,len(run))+1):
                    segment=run[-size:]
                    if np.dot(segment[0][4],t) < .72:
                        continue
                    middle=np.mean([v[3] for v in segment],axis=0)
                    score=np.linalg.norm(middle-hubs[hub]) + (8-size)*33
                    candidates.append((score,hub,segment))
    candidates.sort(key=lambda item:item[0])
    counts=[0]*len(hubs)
    used=[]
    rows=[]
    # Fill each district in turn. A buffer at row ends leaves breaks between blocks.
    while len(used)<len(houses):
        left=len(houses)-len(used)
        blocked=STRtree([v[-2].buffer(3) for v in used])
        chosen=None
        for hub in sorted(range(len(hubs)),key=counts.__getitem__):
            for _,district,row in candidates:
                if district!=hub or len(row)>left or left-len(row) in (1,2):
                    continue
                if any(blocked.query(v[-2],predicate="intersects").size for v in row):
                    continue
                if any(a[-2].intersects(b[-2]) for a,b in zip(row,row[1:])):
                    continue
                chosen=row
                break
            if chosen:
                break
        if chosen is None:
            raise ValueError(f"Only {len(used)} of {len(houses)} housing sites fit")
        rows.append(chosen)
        used.extend(chosen)
        counts[chosen[0][0]]+=len(chosen)

    old=np.array([h["c"] for h in houses])
    new=np.array([v[3]+v[5]*15 for v in used])
    house_indices,site_indices=linear_sum_assignment(cdist(old,new,"sqeuclidean"))
    row_numbers=[i for i,row in enumerate(rows,1) for _ in row]
    for i,j in zip(house_indices,site_indices):
        house=houses[i]
        hub,name,kind,c,t,n,side,_,_=used[j]
        hd,hw=house["hd"],house["hw"]
        site=c+n*(WIDTH[kind]/2 + WALK[kind] + 5 + hd)
        build=rectangle(site,t,n,2*hw+.32,2*hd+.32)
        ground=max(relief(x,y) for x,y in list(build.exterior.coords)[:-1])+.035
        old_t,old_n=house["t"],house["n"]
        old_side=1 if old_t[0]*old_n[1]>old_t[1]*old_n[0] else -1
        turn=math.atan2(t[1],t[0])-math.atan2(old_t[1],old_t[0])
        if old_side!=side:
            turn+=math.pi
        placement=plan["placements"][house["key"]]
        placement["delta"]+=math.degrees(math.atan2(math.sin(turn),math.cos(turn)))
        house.update(road=name,c=site.tolist(),t=t.tolist(),n=n.tolist(),base=ground,
                     row=row_numbers[j],district=hub+1)
        placement.update(road=name,xy=house["c"],base=ground)
        foundation(geometry,site,t,n,hw,hd,ground,build,c,kind,"home")


def plan(work):
    src = json.loads((work / "inventory.json").read_text())
    rng = random.Random(795321)
    g = Geometry()
    terrain(g)
    roads = routes()
    paving = unary_union([l.buffer(WIDTH[k] / 2, quad_segs=6) for _, k, l in roads])
    street = unary_union(
        [l.buffer(WIDTH[k] / 2 + WALK[k], quad_segs=6) for _, k, l in roads]
    )
    # A continuous pavement with no doubling at junctions.
    g.footprint("TownStreets", "Road", paving, lambda x, y: relief(x, y) + 0.105)
    g.footprint(
        "TownStreets",
        "Apron",
        street.difference(paving),
        lambda x, y: relief(x, y) + 0.14,
    )
    for name, k, line in roads:
        if k != "main":
            continue
        for d in np.arange(12, line.length - 8, 20):
            c, t, n = frame(line, d, 1)
            if any(l.distance(Point(c)) < 11 for nn, _, l in roads if nn != name):
                continue
            g.footprint(
                "TownStreets",
                "Stripe",
                rectangle(c, t, n, 3.3, 0.12),
                lambda x, y: relief(x, y) + 0.122,
            )
    kit = (
        "HouseHarborTerrace",
        "HouseDormerPair",
        "HouseHippedCottage",
        "ShopBakery",
        "ShopChandlery",
    )
    buildings = [
        r
        for r in src
        if r["folder"] == "Buildings"
        or (r["folder"] == "StreetFrontages" and r["mesh"] in kit)
    ]

    def sort_key(r):
        label = r["label"]
        if label in ("Building_071", "Building_085"):
            return (0, label)
        return (1 if r["folder"] == "Buildings" else 2, -max(r["e"][:2]), r["key"])

    buildings.sort(key=sort_key)
    lots = []
    occupied = []
    decor = []
    placements = {}
    slots = []
    for i, (name, k, line) in enumerate(roads):
        for d in np.arange(rng.uniform(12, 17), line.length - 11, 7.7):
            for side in (-1, 1):
                c, t, n = frame(line, d, side)
                if max(abs(c[0]), abs(c[1])) > 940 or c[1] < -135:
                    continue
                if k == "dock" and n[1] < -0.3:
                    continue
                slots.append((name, k, c, t, n, side))

    # The ferry approaches and quay have a different land use from the inland
    # streets. Use broad neighbourhood centres so the spaces between them can
    # carry fields and gardens; along a street, buildings still face the kerb.
    neighbourhoods = [
        (-610, 248, 230, 110, 0.74), (-15, 265, 215, 115, 0.80),
        (-302, 423, 235, 130, 1.00), (410, 427, 190, 135, 0.92),
        (-690, 710, 190, 125, 1.10), (-390, 864, 195, 110, 1.30),
        (78, 660, 170, 116, 1.00), (552, 817, 230, 105, 1.38),
    ]
    housing = []
    for _, kind, c, _, _, _ in slots:
        x, y = c
        density = sum(
            strength * math.exp(-((x - cx) / wx) ** 2 - ((y - cy) / wy) ** 2)
            for cx, cy, wx, wy, strength in neighbourhoods
        )
        housing.append((0.07 + density) if y >= 105 and kind != "dock" else 0.0)

    street_prepared = shapely.prepared.prep(street)

    tree = None
    indexed = 0

    def clashes(poly):
        if tree is not None and len(tree.query(poly, predicate="intersects")):
            return True
        return any(v.intersects(poly) for v in occupied[indexed:])

    for r in buildings:
        num = int(r["label"][-3:]) if r["folder"] == "Buildings" else 0
        k = (
            "works"
            if 95 <= num <= 110
            else (
                "shop"
                if 71 <= num <= 94 or r["mesh"].startswith("Shop")
                else "civic" if num >= 111 else "home"
            )
        )
        hw, hd = r["e"][:2]
        ranked = []
        for j, (name, kind, c, t, n, side) in enumerate(slots):
            if k == "works":
                if kind not in ("dock", "main") or c[0] > 100 or c[1] > 50:
                    continue
                score = 4 if kind == "dock" else 0.7
            elif k == "shop":
                if kind not in ("old", "main", "village"):
                    continue
                score = housing[j] * (1.7 if kind == "old" else 0.85)
                if num == 71:
                    score = (
                        math.exp(-(((c[0] + 685) / 38) ** 2) - ((c[1] + 30) / 18) ** 2)
                        if name == "Harbour approach" and side == -1
                        else 0
                    )
                if num == 85:
                    score = (
                        math.exp(-(((c[0] + 420) / 45) ** 2) - ((c[1] - 28) / 35) ** 2)
                        if name == "Harbour approach"
                        else 0
                    )
            else:
                score = housing[j]
                if k == "civic" and kind == "court":
                    continue
            if score > 1e-8:
                ranked.append((-math.log(max(1e-12, rng.random())) / score, j))
        ranked.sort()
        for _, j in ranked:
            name, kind, c, t, n, side = slots[j]
            gap = (
                rng.uniform(0.8, 2.6)
                if k == "shop" or kind == "old"
                else rng.uniform(3.3, 10.0)
            )
            off = WIDTH[kind] / 2 + WALK[kind] + hd + gap
            site = c + n * off
            build = rectangle(site, t, n, 2 * hw + 0.32, 2 * hd + 0.32)
            side_clearance = 3.0 if k in ("shop", "works") else 8.0
            reserve = rectangle(site, t, n, 2 * hw + side_clearance, 2 * hd + 2.8)
            if (
                street_prepared.intersects(reserve)
                or clashes(reserve)
                or (site[1] < -126)
                or max(abs(site)) > 978
            ):
                continue
            break
        else:
            raise RuntimeError(("no site", r["label"], k, len(lots)))
        occupied.append(reserve)
        if num == 71:
            # The auction forecourt opens across the Strand, giving vans and inspectors a clear view.
            court = Polygon(
                [
                    site + t * x - n * y
                    for x, y in (
                        (-9, hd),
                        (8, hd),
                        (12, 26),
                        (7, 42),
                        (-9, 38),
                        (-13, 22),
                    )
                ]
            )
            occupied.append(court)
            g.footprint(
                "TownLots",
                "CompactedQuayGravel",
                court.difference(street),
                lambda x, y: relief(x, y) + 0.072,
            )
        if len(occupied) - indexed >= 20:
            tree = STRtree(occupied)
            indexed = len(occupied)
        x0, y0 = r["c"][:2]
        if k == "works":
            oldside = 1
        else:
            base = min((85, 290, 495, 700, 905), key=lambda a: abs(a - y0))
            oldside = 1 if y0 > base else -1
        angle = math.degrees(math.atan2(t[1], t[0])) + (180 if side != oldside else 0)
        ground = max(relief(x, y) for x, y in list(build.exterior.coords)[:-1]) + 0.035
        placements[r["key"]] = {
            "xy": site.tolist(),
            "base": ground,
            "delta": angle,
            "road": name,
        }
        lots.append(
            dict(
                key=r["key"],
                label=r["label"],
                kind=k,
                road=name,
                c=site.tolist(),
                n=n.tolist(),
                t=t.tolist(),
                hw=hw,
                hd=hd,
                base=ground,
            )
        )
        if k != "home":
            foundation(g, site, t, n, hw, hd, ground, build, c, kind, k)
        front = site - n * hd
        if r["label"] in ("Building_071", "Building_085"):
            decor.append(["QuayFishStall", *(front - n * 1.6 + t * 3), angle, 0])
        if r["mesh"] == "ShopBakery":
            decor.append(["BakeryTables", *(front - n * 1.4), angle, 0])
    occupied_tree = STRtree(occupied)

    def free(shape):
        return not street_prepared.intersects(shape) and not len(
            occupied_tree.query(shape, predicate="intersects")
        )

    # Allocate rear gardens only where the adjoining plot and lane leave room.
    garden_boxes = []
    for L in lots:
        c, t, n = (np.array(L[k]) for k in ("c", "t", "n"))
        hw, hd = L["hw"], L["hd"]
        if L["kind"] == "works":
            hardstand = rectangle(
                c + n * 1.8, t, n, hw * 2 + 8, hd * 2 + 11
            ).difference(street)
            g.footprint(
                "TownLots",
                "CompactedQuayGravel",
                hardstand,
                lambda x, y: relief(x, y) + 0.065,
            )
            for j, m in enumerate(
                ("SM_Crate_Stack_02", "SM_Pallet_3_Stack", "SM_Container")
            ):
                p = c + n * (hd + 4.5) + t * (j - 1) * 5
                if free(Point(p).buffer(3.3)):
                    decor.append(
                        [m, *p, math.degrees(math.atan2(t[1], t[0])), (2.1, 2, 5.9)[j]]
                    )
            continue
        if L["kind"] != "home":
            continue
        depth = rng.uniform(6.7, 12)
        rear = c + n * (hd + depth / 2 + 0.5)
        pocket = rectangle(rear, t, n, min(hw * 2, 14), depth)
        if not free(pocket) or any(pocket.intersects(v) for v in garden_boxes):
            continue
        garden_boxes.append(pocket)
        bearing = math.degrees(math.atan2(t[1], t[0]))
        if rng.random() < 0.48 and hw > 3.6:
            decor.append(
                [
                    "GardenWorkshop",
                    *(rear + t * (min(hw, 6) - 2.6)),
                    bearing + rng.uniform(-9, 9),
                    0,
                ]
            )
        if hw > 6 and rng.random() < 0.68:
            decor.append(["GardenRail", *(c + n * (hd + depth + 0.55)), bearing, 0])
        decor.append(
            [
                "tree",
                *(rear - t * (min(hw, 6) - 1.9)),
                rng.uniform(0, 360),
                rng.uniform(2.6, 4.2),
            ]
        )
    vehicles = [r for r in src if r["folder"] == "Vehicles"]
    used = []
    for r in vehicles:
        if r["c"][1] < -140:
            continue
        j = int(r["label"][-3:])
        for attempt in range(500):
            name, kind, l = rng.choice(roads)
            if kind == "dock" and j % 4:
                continue
            d = rng.uniform(14, l.length - 14)
            p, t, n = frame(l, d, rng.choice((-1, 1)))
            if max(abs(p)) > 955 or any(
                q.distance(Point(p)) < 13 for nn, _, q in roads if nn != name
            ):
                continue
            center = p + n * (WIDTH[kind] / 2 - min(r["e"][:2]) - 0.38)
            if any(np.linalg.norm(center - other) < 12 for other in used):
                continue
            used.append(center)
            break
        else:
            raise RuntimeError(r["label"])
        placements[r["key"]] = {
            "xy": center.tolist(),
            "base": relief(*center) + 0.107,
            "delta": math.degrees(math.atan2(t[1], t[0])),
            "road": name,
        }
    # Field enclosures and small stands of trees give the transition to open land a visible scale.
    seeds = np.array(
        [
            (x + rng.uniform(-80, 80), y + rng.uniform(-64, 64))
            for x in range(-2900, 2901, 218)
            for y in range(30, 2940, 165)
        ]
    )
    vor = Voronoi(seeds)
    field_limit = box(-2240, 120, 2280, 2270)
    town = unary_union(occupied + garden_boxes).buffer(16)
    exclude = town.union(street.buffer(7))
    for ix, region in enumerate(vor.point_region):
        ids = vor.regions[region]
        if not ids or -1 in ids:
            continue
        cell = Polygon(vor.vertices[ids]).intersection(field_limit)
        if cell.is_empty or cell.area < 2300 or cell.intersects(exclude):
            continue
        field = cell.buffer(-rng.uniform(1.4, 3.2))
        if field.is_empty:
            continue
        if ix % 3:
            g.footprint(
                "FieldParcels",
                ("MeadowPasture", "MeadowStubble", "MeadowHay")[ix % 3],
                field,
                lambda x, y: relief(x, y) + 0.11,
                22,
            )
        if ix % 3 != 1:
            perimeter = field.exterior
            for s in np.arange(
                rng.uniform(0, 13), perimeter.length * 0.68, rng.uniform(12, 25)
            ):
                p = xy(perimeter.interpolate(s))
                decor.append(["tree", *p, rng.uniform(0, 360), rng.uniform(3, 7)])
    for cx, cy in [
        (-1020, 612),
        (-330, 535),
        (250, 715),
        (600, 321),
        (795, 72),
        (-1290, 72),
        (1140, 350),
        (530, 1270),
        (-840, 1100),
    ]:
        for _ in range(40):
            p = (rng.gauss(cx, 46), rng.gauss(cy, 31))
            if free(Point(p).buffer(6)):
                decor.append(["tree", *p, rng.uniform(0, 360), rng.uniform(4, 8)])
    for item in decor:
        item.append(relief(item[1], item[2]))
    data = dict(
        quay_props=working_quay(g, src),
        placements=placements,
        lots=lots,
        decor=decor,
        roads=[dict(name=a, kind=b, xy=list(c.coords)) for a, b, c in roads],
    )
    trim_layout(data)
    arrange_residential_rows(data, g, street)
    from harbor_perimeter import survey_perimeter
    data["water_bounds_m"] = survey_perimeter(g.meshes, MODEL_BOUNDARY)
    (work / "plan.json").write_text(json.dumps(data, separators=(",", ":")))
    (work / "geometry.json").write_text(json.dumps(g.meshes, separators=(",", ":")))
    print(
        len(lots),
        len(placements),
        len(data["decor"]),
        {a: sum(map(len, b.values())) for a, b in g.meshes.items()},
    )


if __name__ == "__main__":
    plan(Path(sys.argv[1]))
