"""洞口候选必须保留横梁、无效深度和画外未知区域。"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from perception.opening_geometry import opening_candidates


def perspective_range(planar):
    h,w=planar.shape
    yy,xx=np.mgrid[:h,:w]
    return planar*np.sqrt(1+((xx+.5-w/2)/(w/2))**2+((yy+.5-h/2)/(w/2))**2)


class OpeningGeometryTests(unittest.TestCase):
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
