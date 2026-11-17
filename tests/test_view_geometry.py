"""Conservative absent-template evidence and training-image redaction."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common.view_geometry import box_maybe_visible, projected_box, redact_rgb
from capture.collect_target_pilot import plan_views


class ViewGeometryTests(unittest.TestCase):
    def test_near_plane_and_frame_crossings_cannot_be_declared_absent(self):
        rotation=np.eye(3);position=np.zeros(3)
        self.assertTrue(box_maybe_visible([[5,-1,-1],[6,1,1]],position,rotation,90,16/9))
        self.assertFalse(box_maybe_visible([[-6,-1,-1],[-5,1,1]],position,rotation,90,16/9))
        self.assertTrue(box_maybe_visible([[-1,-10,-10],[2,10,10]],position,rotation,90,16/9))
        self.assertEqual(projected_box([[-1,-10,-10],[2,10,10]],position,rotation,90,64,36),[0,0,64,36])
        self.assertFalse(box_maybe_visible([[5,10,-1],[6,11,1]],position,rotation,90,16/9))

    def test_redaction_erases_only_the_marked_pixels_without_mutating_raw_rgb(self):
        rgb=np.full((4,4,3),7,np.uint8);mask=np.array([[1,0],[0,0]],np.uint8)
        result=redact_rgb(rgb,mask)
        np.testing.assert_array_equal(result[:2,:2],np.broadcast_to([124,116,104],(2,2,3)))
        self.assertTrue((rgb==7).all());self.assertTrue((result[2:]==7).all())

    def test_partition_sampling_is_balanced_repeatable_and_does_not_redact_test(self):
        from copy import deepcopy
        base={'id':'train','index':1,'category':'aircraft','reference_ned_m':[0,0,-3],
              'region':{'forward_ne':[1,0]},'bounds_ned_m':[[-3,-3,-5],[3,3,0]],'appearance_group':'a'}
        other=deepcopy(base);other.update(id='test',index=2,appearance_group='b',bounds_ned_m=[[20,20,-5],[25,25,0]])
        catalog={'targets':[base,other],'primary_protocol':'appearance','splits':{'appearance':{'train':['train'],'validation':[],'test':['test']}}}
        protocol={'target_indices':[1,2],'partitions':['train','test'],'sampling':'stratified_random','seed':7,
                  'samples_per_partition_target':{'train':8,'test':8},'range_strata_m':[[12,30],[30,60]],
                  'bearing_ranges_deg':{'aircraft':[-90,90]},'heading_offsets_deg':[-25,25],
                  'vertical_jitter_ratio':[-.15,.15],'reference_height_offsets_m':{'aircraft':[0]},
                  'camera_pitch_deg':35,'horizontal_fov_deg':90,'rgb_wh':[896,504],'depth_wh':[64,36],
                  'redact_later_partition_appearances':True,'redact_other_training_instances':True}
        first=plan_views(catalog,protocol);self.assertEqual(first,plan_views(catalog,protocol))
        self.assertEqual(len(first),16)
        self.assertEqual(sum(c['range_stratum']==0 for c in first),8)
        self.assertTrue(any(abs(c['heading_offset_deg'])>1 for c in first))
        self.assertTrue(all(not c['ignored_depth_boxes'] for c in first if c['partition']=='test'))


if __name__=='__main__':unittest.main()
