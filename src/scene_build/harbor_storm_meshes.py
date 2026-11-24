"""Blender 3.6. Export wind-damaged building variants and grounded debris assemblies."""
import math
import random
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).parent))
from harbor_meshes import Model, PAINT, STEEL, WOOD, DARK, OUT
from harbor_street_meshes import house, beam, CREAM, SLATE, MOSS


def broken_roof(side, severe):
    def build(m, cx, cy, width, depth, eave, rise, color, hip):
        w, d = width/2, depth/2
        inset = min(d*.8, w*.65) if hip else 0
        def at(x, y, dz=0):
            return (x, y, eave + rise*(1-abs(y)/d) + dz)
        def face(points, tint):
            normal=(Vector(points[1])-Vector(points[0])).cross(Vector(points[2])-Vector(points[0]))
            m.part(points,[tuple(range(len(points))) if normal.z>=0 else tuple(reversed(range(len(points))))],tint)
        def inner(x):
            return d*(1-min(1,(w-abs(x))/inset)) if hip else 0

        # Keep the opposite slope, ridge and hip ends. The attic floor is visible through real openings.
        face([at(-w,-side*d),at(w,-side*d),at(w-inset,0),at(-w+inset,0)],color)
        for sign in (-1,1):
            a,b,c=at(sign*w,-d),at(sign*w,d),at(sign*(w-inset),0)
            m.part([a,b,c],[(0,1,2),(2,1,0)],color if hip else CREAM)
            for tip in (a,b):
                beam(m,tip,c,.10,.13,STEEL)
        m.box((0,0,eave+.025),(width-.96,depth-.96,.04),DARK)
        beam(m,at(-w+inset,0),at(w-inset,0),.14,.14,STEEL)
        beam(m,at(-w,-side*d),at(w,-side*d),.11,.13,STEEL)

        bays=max(10,math.ceil(width/.78))
        for j in range(bays):
            x0=-w+width*j/bays; x1=-w+width*(j+1)/bays
            # Use varying lengths at torn seams; keep panels on both sides of the breach.
            mid=(j+.5)/bays
            start,end=(.12,.81) if severe else (.48,.85)
            loss=0
            if start<mid<end:
                loss=(.72 if severe else .38)+.19*math.sin(2.1*j+.3)
                if min(mid-start,end-mid)<.09:loss*=.67
            y0,y1=inner(x0),inner(x1)
            a,b=y0+(d-y0)*(1-loss),y1+(d-y1)*(1-loss)
            if min(d-y0,d-y1)<.001 and max(d-y0,d-y1)<.001:
                continue
            fold=(.19 if severe else .31) if loss and j%3!=0 else 0
            face([at(x0,side*y0),at(x1,side*y1),at(x1,side*b,fold*.7),at(x0,side*a,fold)],color)
            if a>y0+.12:
                beam(m,at(x0,side*y0,.026),at(x0,side*a,fold+.026),.033,.046,color)
            if loss:
                # Lower framing, with a few laths still fastened between neighbouring rafters.
                x=(x0+x1)/2
                beam(m,at(x,side*inner(x),-.18),at(x,side*(d-.09),-.18),.13,.20,WOOD)
                for k in range(1,5):
                    f=k/5
                    if f<1-loss or (j+k)%4==0:continue
                    left=y0+(d-y0)*f;right=y1+(d-y1)*f
                    stop=x1 if (j+2*k)%5 else x1-.26*(x1-x0)
                    beam(m,at(x0,side*left,-.10),at(stop,side*right,-.10),.085,.065,WOOD)
            else:
                beam(m,at(x0,side*d,-.02),at(x1,side*d,-.02),.10,.12,STEEL)
            x=x0
            if d-inner(x)>.1:
                beam(m,at(x,-side*inner(x),.024),at(x,-side*d,.024),.034,.043,color)
    return build


def branch(m, a, b, r0, r1, rng):
    a,b=Vector(a),Vector(b)
    axis=(b-a).normalized()
    radial=axis.cross(Vector((0,0,1)) if abs(axis.z)<.9 else Vector((0,1,0))).normalized()
    tangent=axis.cross(radial)
    vs=[];sides=8
    for pt,r in ((a,r0),(b,r1)):
        for j in range(sides):
            angle=j*math.tau/sides
            vs.append(pt+(radial*math.cos(angle)+tangent*math.sin(angle))*r*rng.uniform(.91,1.09))
    for j in range(sides):
        k=(j+1)%sides
        m.part([vs[j],vs[k],vs[k+sides],vs[j+sides]],[(0,1,2,3)],(.22+.018*(j%4),.18,.13))
    m.part(vs[sides:], [tuple(range(sides))],WOOD)
    m.part(vs[:sides], [tuple(reversed(range(sides)))],WOOD)


def sheet(m, x, y, length, width, angle, kink, tint, z=0):
    # Fluted metal, folded around a transverse crease, with an uneven loose edge.
    a=math.radians(angle); count=max(12,round(width/.052))
    def p(t,j):
        v=width*(j/count-.5)
        u=(t-.5)*length+(0.07*math.sin(j*1.3) if t==1 else 0)
        h=.065+.032*math.cos(j*math.pi/2)+max(0,t-.63)*kink*(1+.3*v)
        return (x+math.cos(a)*u-math.sin(a)*v,y+math.sin(a)*u+math.cos(a)*v,z+h)
    for lo,hi in ((0,.63),(.63,1)):
        for j in range(count):
            pts=[p(lo,j),p(hi,j),p(hi,j+1),p(lo,j+1)]
            m.part(pts,[(0,1,2,3),(3,2,1,0)],tint)


def debris():
    for i in range(3):
        rng=random.Random(142+i);m=Model()
        sheet(m,-.8,.2,2.2+.25*i,.93,9+i*21,.65+.3*i,(.42,.47,.46))
        sheet(m,1.65,-.32,1.35,.72,-31+i*15,.42,(.34,.39,.40))
        for _ in range(4):
            length=rng.uniform(.48,1.9)
            m.box((rng.uniform(-2.1,2),rng.uniform(.82,1.7),.081),
                  (length,.10,.09),WOOD,(0,0,rng.uniform(-45,55)))
        m.save("SheetScatter"+str(i))
    for i in range(2):
        m=Model();rng=random.Random(592+i)
        spine=[(-4.9,0,.28),(-2.1,.13,.36),(.6,-.1,.66),(2.9,.54,.48),(5.05,.17,.22)]
        for j,(a,b) in enumerate(zip(spine,spine[1:])):
            branch(m,a,b,.28-.051*j,.23-.048*j,rng)
        for j in range(12):
            a=Vector(spine[1+j%3]);s=(-1)**j
            b=a+Vector((rng.uniform(.7,1.7),s*rng.uniform(.8,1.8),rng.uniform(-.12,.5)))
            c=b+Vector((rng.uniform(.45,1),s*.65,-min(.45,b.z-.12)))
            branch(m,a,b,.095,.048,rng);branch(m,b,c,.052,.018,rng)
            for k in range(3):
                origin=b.lerp(c,k/3)
                branch(m,origin,origin+Vector((.38,s*rng.uniform(-.5,.6),rng.uniform(0,.52))),.028,.008,rng)
        for j in range(5):
            branch(m,spine[0],(-5.3-rng.random()*.32,(j-2)*.074,.19+rng.random()*.18),.065,.012,rng)
        m.save("BrokenCrown"+str(i))
    for i in range(2):
        m=Model();rng=random.Random(827+i)
        sheet(m,-1,-.6,3.4,1.22,18+17*i,1.25,(.46,.48,.46))
        for j in range(9):
            x=rng.uniform(-2.5,2.1);y=rng.uniform(.6,2.0)
            m.box((x,y,.09+j%3*.048),(rng.uniform(.9,2.5),.15,.095),WOOD,(0,2,j*13-36))
        # A broken pallet with two runners still holding one end together.
        for y in (-.43,.43):m.box((2.6,y,.11),(1.6,.11,.16),WOOD)
        for j in range(4):m.box((2.0+.28*j,0,.24),(.18,1.16,.09),WOOD,(0,0,2*j))
        m.save("QuayWreckage"+str(i))
    m=Model();rng=random.Random(748)
    for j in range(17):
        x,y=rng.uniform(-3.6,3),rng.uniform(-1,1)
        branch(m,(x,y,.12),(x+rng.uniform(.6,2.1),y+rng.uniform(-.45,.8),.10),
               rng.uniform(.07,.16),.04,rng)
    m.save("TideWrack")


def village_breach(source):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source),use_anim=False)
    obj,=[o for o in bpy.context.scene.objects if o.type=="MESH"]
    mesh=obj.data
    mesh.transform(obj.matrix_world);obj.matrix_world=Matrix.Identity(4)
    mesh.calc_loop_triangles()
    tree=BVHTree.FromPolygons([v.co.copy() for v in mesh.vertices],
                              [t.vertices[:] for t in mesh.loop_triangles],all_triangles=True)
    def z(x,y):
        hit=tree.ray_cast((x,y,20),(0,0,-1))[0]
        if hit is None or hit.z<5:raise ValueError((x,y,hit))
        return hit.z
    bm=bmesh.new();bm.from_mesh(mesh)
    x0,x1=-3.05,.24
    bands=[(-2.8,-1.9,x0+.6),(-1.9,.8,x0),(.8,1.85,x0+.3),(1.85,2.55,x0+1.1)]
    cuts=[(x,0) for x in {x1,*[a for _,_,a in bands]}]+[(y,1) for y in sorted({v for lo,hi,_ in bands for v in (lo,hi)})]
    for value,axis in cuts:
        point=Vector((0,0,0));point[axis]=value;normal=point.copy();normal[axis]=1
        bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
                              plane_co=point,plane_no=normal,dist=.00001)
    remove=[]
    for face in bm.faces:
        x,y,h=face.calc_center_median()
        if h>5.4 and any(lo<y<hi and left<x<x1 for lo,hi,left in bands):
            remove.append(face)
    assert remove
    bmesh.ops.delete(bm,geom=remove,context='FACES');bm.to_mesh(mesh);bm.free()
    # Fit the exposed framing to the textured source roof rather than its world-aligned bounding box.
    frame=Model()
    for lo,hi,left in bands:
        pts=[(x,y,z(x,y)-.58) for x,y in ((left,lo),(x1,lo),(x1,hi),(left,hi))]
        frame.part(pts,[(0,1,2,3)],DARK)
        for j in range(max(1,round((hi-lo)/.67))):
            y=lo+.24+j*.67
            beam(frame,(left-.12,y,z(left,y)-.20),(x1+.08,y,z(x1,y)-.20),.135,.19,WOOD)
        for j in range(2):
            x=left+(j+.75)*(x1-left)/2.6
            beam(frame,(x,lo,z(x,lo)-.11),(x,hi-.14*j,z(x,hi)-.11),.08,.067,WOOD)
    frm=bpy.data.meshes.new("RoofFrame");frm.from_pydata(frame.points,[],frame.faces);frm.update()
    for mat in ("Timber","Rubber"):frm.materials.append(bpy.data.materials.new(mat))
    for poly,col in zip(frm.polygons,frame.colors):poly.material_index=int(col==DARK)
    uv=frm.uv_layers.new(name="FrameUV")
    for poly in frm.polygons:
        for idx in poly.loop_indices:
            v=frm.vertices[frm.loops[idx].vertex_index].co
            uv.data[idx].uv=(v.x*.4,v.y*.4+v.z*.25)
    ob=bpy.data.objects.new("RoofFrame",frm);bpy.context.collection.objects.link(ob);ob.select_set(True)
    obj.select_set(True);bpy.context.view_layer.objects.active=obj;bpy.ops.object.join()
    obj.name="VillageRoofBreach"
    bpy.ops.export_scene.fbx(filepath=str(OUT/"VillageRoofBreach.fbx"),use_selection=True,
                             axis_forward="-Y",axis_up="Z",mesh_smooth_type="FACE",bake_anim=False)


def tug_mast(source):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source),use_anim=False)
    obj,=[o for o in bpy.context.scene.objects if o.type=="MESH"]
    mesh=obj.data
    mesh.transform(obj.matrix_world);obj.matrix_world=Matrix.Identity(4)
    # The mast stands above the wheelhouse. Add vertices at its failed joint so the lower post stays fixed.
    hinge=15.78
    bm=bmesh.new();bm.from_mesh(mesh)
    for height in (hinge,hinge+.22):
        bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
                               plane_co=(0,0,height),plane_no=(0,0,1),dist=.00001)
    bm.to_mesh(mesh);bm.free()
    for vert in mesh.vertices:
        x,y,z=vert.co
        if z<=hinge:continue
        angle=math.radians(69)*min(1,(z-hinge)/.22)
        vert.co.y=y*math.cos(angle)-(z-hinge)*math.sin(angle)
        vert.co.z=hinge+y*math.sin(angle)+(z-hinge)*math.cos(angle)
    mesh.update()
    obj.name="TugBentMast"
    bpy.ops.export_scene.fbx(filepath=str(OUT/"TugBentMast.fbx"),use_selection=True,
                             axis_forward="-Y",axis_up="Z",mesh_smooth_type="FACE",bake_anim=False)


if __name__=="__main__":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for name,units,h,hip,wall,tint in [
        ("HarborTerrace",3,5.95,False,CREAM,SLATE),("DormerPair",2,5.55,False,MOSS,PAINT[0]),
        ("HippedCottage",1,3.24,True,(.79,.70,.60),PAINT[3])]:
        for severity in ("Peeled","Stripped"):
            for side,letter in ((1,"A"),(-1,"B")):
                house(name+severity+letter,units,h,hip,wall,tint,broken_roof(side,severity=="Stripped"))
    debris()
    source=Path(sys.argv[sys.argv.index("--")+2])
    village_breach(source/"village-house/village_house_Storm.fbx")
    tug_mast(source/"tugboat/tugboat_Storm.fbx")
