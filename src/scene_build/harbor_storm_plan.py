"""Place static wind and retreating-water evidence in UE survey metres."""
import json
import math
import random
import sys
from pathlib import Path

from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree


def plan_storm(plan, inventory, catalog):
    rng = random.Random(11944)
    inv = {v["key"]: v for v in inventory}
    lots = plan["lots"]
    protected = []
    for h in lots:
        c, t, n = h["c"], h["t"], h["n"]
        protected.append(Polygon([(c[0]+a*t[0]+b*n[0], c[1]+a*t[1]+b*n[1])
                                  for a,b in ((-h["hw"],-h["hd"]),(h["hw"],-h["hd"]),
                                              (h["hw"],h["hd"]),(-h["hw"],h["hd"]))]).buffer(.8))
    for row in catalog:
        if row["category"] != "Building":
            radius = max(4, inv.get(row.get("ue_actor_name"),{}).get("e",[0,0])[0]) + 1.8
            protected.append(Point(*row["xyz_m"][:2]).buffer(radius))
    protected += [Point(x,y).buffer(size*.45) for kind,x,y,_,size,_ in plan["decor"] if kind=="tree"]
    obstacles = STRtree(protected)
    roads = STRtree([LineString(r["xy"]).buffer(5.8) for r in plan["roads"]])
    placed = []
    zones = [(-665,235,185,150),(-134,318,180,145),(384,364,165,135),(-500,805,115,100)]
    roofs = []
    for h in lots:
        base = inv[h["key"]]["mesh"]
        if h["kind"] != "home" or not base.startswith("House") or not h["label"].startswith("StreetFrontages"):
            continue
        x,y = h["c"]
        exposure=max(math.exp(-((x-cx)/sx)**2-((y-cy)/sy)**2) for cx,cy,sx,sy in zones)
        if rng.random() >= .10 + .72*exposure:
            continue
        severity = "Stripped" if rng.random() < .22+.50*exposure else "Peeled"
        yaw=math.radians(inv[h["key"]]["yaw"]+plan["placements"][h["key"]]["delta"])
        # Blender +Y becomes Unreal -Y. Select the roof slope facing incoming marine wind.
        side="A" if math.cos(yaw)-.28*math.sin(yaw)>0 else "B"
        roofs.append(dict(label=h["label"],mesh=base+severity+side,severity=severity,
                          xy=[x,y],base=h["base"],road=h["road"],district=h["district"]))
    def scatter(kind, anchor, radius, offset, source, surface="land", avoid_road=True):
        for _ in range(30):
            x=anchor[0]+rng.uniform(-offset,offset)+3
            y=anchor[1]+rng.uniform(1,offset)
            point=Point(x,y).buffer(radius)
            if obstacles.query(point,predicate="intersects").size or any(point.intersects(q) for q in placed):
                continue
            if avoid_road and roads.query(point,predicate="intersects").size:
                continue
            placed.append(point)
            props.append(dict(mesh=kind,xy=[round(x,3),round(y,3)],yaw=round(rng.uniform(52,98),2),
                              radius=radius,source=source,surface=surface))
            return True
        return False
    props=[]
    for i,r in enumerate(roofs):
        if r["severity"]=="Stripped" or i%4==0:
            scatter("SheetScatter"+str(i%3),r["xy"],4.8,22,r["label"])
        if r["severity"]=="Stripped" and i%3==0:
            scatter("BrokenCrown"+str(i%2),r["xy"],6.4,36,r["label"])
    for row in catalog:
        if row["id"] in ("Building_001","Building_013","Building_036","Building_095","Building_098","Building_112"):
            for i in range(2):
                scatter("SheetScatter"+str(i),row["xyz_m"],4.8,26,row["id"])
    # Quay apron, with clear space around the piers, fish market and player start.
    for j,(x,y) in enumerate([(-819,-218),(-747,-228),(-680,-221),(-620,-216),(-535,-224),
                              (-435,-219),(-336,-210),(-273,-229),(-175,-218),(-93,-200)]):
        scatter("QuayWreckage"+str(j%2),(x,y),5.6,12,"WorkingQuay","quay")
    for v in catalog:
        if v["category"]=="Vessel" and v["state"]=="aground":
            scatter("TideWrack",v["xyz_m"],5.5,23,v["id"],"mud",False)
    return dict(roofs=roofs,props=props,wind_towards_xy=[.28,1.0])


if __name__=="__main__":
    root=Path(sys.argv[1])
    values=[json.loads((root/(key+".json")).read_text()) for key in ("plan","inventory","catalog")]
    (root/"wind-layout.json").write_text(json.dumps(plan_storm(*values),indent=2))
