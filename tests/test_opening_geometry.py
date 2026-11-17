"""洞口候选必须保留横梁、无效深度和画外未知区域。"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from perception.opening_geometry import opening_candidates
from perception.opening_matching import body_constrained_match
from types import SimpleNamespace
from control.run_opening_alignment import horizontal_error


def perspective_range(planar):
    h,w=planar.shape
    yy,xx=np.mgrid[:h,:w]
    return planar*np.sqrt(1+((xx+.5-w/2)/(w/2))**2+((yy+.5-h/2)/(w/2))**2)


class OpeningGeometryTests(unittest.TestCase):
    def test_group_alignment_stays_stable_when_beam_splits_free_gap(self):
        plane=np.full((120,240),10.,np.float32)
        plane[40:80,80:115]=25.;plane[40:80,117:155]=25.;plane[60,115:117]=25.
        rgb=np.zeros((120,240,3),np.uint8)
        fp=SimpleNamespace(query=lambda image:(0.,-.9,1.,.8,None))
        first,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,down=np.array([0.,0.,1.]))
        plane[60,115:117]=10.
        second,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,down=np.array([0.,0.,1.]),previous_box=first['selected']['bbox_depth_xyxy'])
        camera=dict(width=240,height=120,rot_w=np.sqrt(.5),rot_x=0.,rot_y=-np.sqrt(.5),rot_z=0.)
        self.assertLess(np.linalg.norm(horizontal_error(second,camera,True,True)-horizontal_error(first,camera,True,True)),.1)
        self.assertGreater(np.linalg.norm(horizontal_error(second,camera)-horizontal_error(first,camera)),1.)
        self.assertFalse(second['group_alignment_point_is_free_passage_proof'])

    def test_verified_gap_can_continue_at_frame_boundary(self):
        plane=np.full((120,240),10.,np.float32);plane[50:119,80:160]=25.
        rgb=np.zeros((120,240,3),np.uint8)
        fp=SimpleNamespace(query=lambda image:(0.,-.9,1.,.8,None))
        first,_=body_constrained_match(fp,rgb,perspective_range(plane),.2)
        previous=first['selected']['bbox_depth_xyxy']
        plane[119:,80:160]=25.
        initial,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,previous_box=previous)
        self.assertIsNone(initial['selected'])
        tracked,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,previous_box=previous,allow_tracked_gap=True)
        self.assertTrue(tracked['selected']['tracked_boundary_gap'])
        self.assertGreater(tracked['selected']['previous_box_iou'],.98)
        plane[:,160:]=25.  # 并入大片画外地面后失去连续性，不能继续冒认为同一空隙。
        merged,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,previous_box=previous,allow_tracked_gap=True)
        self.assertIsNone(merged['selected'])

    def test_long_opening_uses_visible_geometric_center(self):
        h,w=360,640;y,x=np.mgrid[:h,:w];f=w/2
        rays=np.stack(((x+.5-w/2)/f,(y+.5-h/2)/f,np.ones((h,w))),axis=-1)
        for degrees in (40,60,80):
            with self.subTest(degrees=degrees):
                down=np.array([0.,np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees))])
                roof=rays*(30/(rays@down))[...,None];center=np.array([0.,0.,30/down[2]])
                horizontal=np.cross(down,np.array([1.,0.,0.]))
                hole=(abs(roof[:,:,0])<12)&(abs((roof-center)@horizontal)<3)
                height=np.where(hole,60.,30.)
                ranges=height/(rays@down)*np.linalg.norm(rays,axis=-1)
                candidate=opening_candidates(ranges,down=down)['candidates'][0]
                px,py=candidate['aperture_point_depth_xy']
                self.assertLess(np.linalg.norm(roof[py,px]-center),.4)
                self.assertGreater(candidate['projected_clearance_m'],2.6)

    def test_verified_gap_continues_when_body_point_enters_boundary_connected_floor(self):
        h,w=120,240;plane=np.full((h,w),10.,np.float32)
        plane[:35,80:160]=25.;plane[60:100,80:160]=25.
        rgb=np.zeros((h,w,3),np.uint8);previous=[80,60,160,100]
        fp=SimpleNamespace(query=lambda image:(0.,-.9,1.,.8,None))
        direct,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,previous_box=previous)
        self.assertIsNone(direct['selected'])
        tracked,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,previous_box=previous,allow_tracked_gap=True)
        self.assertIsNotNone(tracked['selected'])
        self.assertEqual(tracked['selected']['identity_association'],'verified_free_gap_continuity')
        fp.query=lambda image:(0.,-.9,1.,.1,None)
        weak,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,previous_box=previous,allow_tracked_gap=True)
        self.assertIsNone(weak['proposed_aperture_point_rgb_xy'])

    def test_selected_free_gap_survives_neighbor_becoming_larger(self):
        h,w=120,240
        plane=np.full((h,w),10.,np.float32);plane[:,230:]=25.
        plane[30:90,30:80]=25.;plane[30:90,82:130]=25.
        rgb=np.zeros((h,w,3),np.uint8)
        fp=SimpleNamespace(query=lambda image:(2*20/w-1,0.,1.,.8,None))
        first,_=body_constrained_match(fp,rgb,perspective_range(plane),.2)
        previous=first['selected']['bbox_depth_xyxy']
        self.assertLess(first['proposed_aperture_point_rgb_xy'][0],80)
        plane[30:90,82:140]=25.
        second,_=body_constrained_match(fp,rgb,perspective_range(plane),.2,previous_box=previous)
        self.assertLess(second['proposed_aperture_point_rgb_xy'][0],80)
        self.assertGreater(second['selected']['previous_box_iou'],.9)

    def test_body_identity_selects_only_its_own_roof_opening(self):
        h,w=120,240
        plane=np.full((h,w),25.,np.float32)
        plane[10:110,10:105]=10.;plane[10:110,135:230]=10.
        plane[35:85,35:80]=25.;plane[35:85,160:205]=25.
        rgb=np.zeros((h,w,3),np.uint8)
        for x in (20,220):
            fp=SimpleNamespace(query=lambda image:(2*x/w-1,0.,1.,.8,None))
            match,_=body_constrained_match(fp,rgb,perspective_range(plane),.2)
            self.assertIsNotNone(match['proposed_aperture_point_rgb_xy'])
            self.assertEqual(match['proposed_aperture_point_rgb_xy'][0]<w/2,x<w/2)
        fp=SimpleNamespace(query=lambda image:(2*20/w-1,0.,1.,.1,None))
        match,_=body_constrained_match(fp,rgb,perspective_range(plane),.2)
        self.assertIsNone(match['proposed_aperture_point_rgb_xy'])

    def test_oblique_circle_has_metric_radius_and_stable_center(self):
        h,w=360,640;y,x=np.mgrid[:h,:w];f=w/2
        rays=np.stack(((x+.5-w/2)/f,(y+.5-h/2)/f,np.ones((h,w))),axis=-1)
        for degrees in (40,60,80):
            with self.subTest(degrees=degrees):
                down=np.array([0.,np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees))])
                roof=rays*(30/(rays@down))[...,None]
                center=np.array([0.,0.,30/down[2]])
                hole=np.linalg.norm(roof-center,axis=-1)<4.
                height=np.where(hole,55.,30.)
                height[:,550:]=55.  # 屋外地面为分层提供足够支持，不能并入闭合洞口。
                ranges=height/(rays@down)*np.linalg.norm(rays,axis=-1)
                result=opening_candidates(ranges,down=down)
                candidate=result['candidates'][0]
                self.assertAlmostEqual(candidate['projected_clearance_m'],4.,delta=.3)
                px,py=candidate['aperture_point_depth_xy']
                self.assertLess(np.linalg.norm(roof[py,px]-center),.35)

    def test_oblique_horizontal_roof_requires_gravity_projection(self):
        h,w=120,160;y,x=np.mgrid[:h,:w];f=w/2
        down=np.array([0.,.5,np.sqrt(.75)])
        height=np.full((h,w),30.,np.float32);height[35:85,55:105]=55.
        dot=down[2]+(y+.5-h/2)/f*down[1]
        ranges=perspective_range(height/dot)
        result=opening_candidates(ranges,down=down)
        self.assertEqual(result['status'],'candidates_found')
        self.assertAlmostEqual(result['near_layer_depth_m'],30.,places=4)
        self.assertAlmostEqual(result['far_layer_depth_m'],55.,places=4)
        point=result['candidates'][0]['aperture_point_depth_xy']
        self.assertTrue(55<=point[0]<105 and 35<=point[1]<85)

    def test_beam_not_filled_into_visible_gap(self):
        plane=np.full((120,160),10.,np.float32)
        plane[25:95,45:115]=25
        plane[25:95,78:82]=10  # 横梁应只在归组时桥接。
        plane[40:43,53:56]=np.inf
        result=opening_candidates(perspective_range(plane))
        self.assertEqual(result['status'],'candidates_found')
        selected=max(result['candidates'],key=lambda c:c['group_area_px'])
        self.assertFalse(selected['visible_gap_mask'][60,80])
        self.assertFalse(selected['visible_gap_mask'][41,54])
        x,y=selected['aperture_point_depth_xy']
        self.assertTrue(selected['visible_gap_mask'][y,x])
        self.assertGreater(selected['projected_clearance_m'],0)

    def test_frame_boundary_is_not_a_closed_opening(self):
        plane=np.full((120,160),10.,np.float32)
        plane[:,0:50]=25
        result=opening_candidates(perspective_range(plane))
        self.assertEqual(result['status'],'no_closed_opening')
        self.assertEqual(result['candidates'],[])

    def test_uniform_and_invalid_depth_do_not_propose_opening(self):
        for plane in [np.full((120,160),20.,np.float32),np.full((120,160),np.inf,np.float32)]:
            self.assertEqual(opening_candidates(perspective_range(plane))['candidates'],[])

    def test_near_border_closed_opening_does_not_expand_to_border(self):
        plane=np.full((120,160),10.,np.float32)
        plane[20:113,45:115]=25
        result=opening_candidates(perspective_range(plane))
        self.assertEqual(result['status'],'candidates_found')
        self.assertTrue(all(c['bbox_depth_xyxy'][3]<120 for c in result['candidates']))

    def test_exterior_ground_does_not_merge_into_nearby_opening(self):
        plane=np.full((120,160),10.,np.float32)
        plane[25:95,100:150]=25
        plane[:,158:]=25
        result=opening_candidates(perspective_range(plane))
        self.assertEqual(result['status'],'candidates_found')
        self.assertEqual(len(result['candidates']),1)
        self.assertLess(result['candidates'][0]['bbox_depth_xyxy'][2],158)


if __name__=='__main__':unittest.main()
