from pathlib import Path
import sys
import unittest
import numpy as np
import cv2
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from perception.template_verification import proposal_box,square_rgb,correspondence_features,rgb_descriptors,rgb_correspondence_features,TemplateVerifier


class TemplateVerificationTests(unittest.TestCase):
    def test_spatially_consistent_matches_beat_scrambled_identical_descriptors(self):
        matrix=np.eye(16,dtype=np.float32);cv2.setRNGSeed(17)
        good,score=correspondence_features(matrix,.02)
        scrambled=matrix[np.random.default_rng(17).permutation(16)];cv2.setRNGSeed(17)
        bad,bad_score=correspondence_features(scrambled,.02)
        self.assertAlmostEqual(good[8],1.);self.assertGreater(score,bad_score)
        self.assertEqual(good[3],bad[3])

    def test_crop_follows_response_component_and_preserves_raw_rgb(self):
        logits=np.full((8,8),-10.,np.float32);logits[2:4,5:7]=2
        box=proposal_box(logits,(100,200),1.5,.12,2)
        self.assertTrue(0<=box[0]<box[2]<=200 and 0<=box[1]<box[3]<=100)
        self.assertTrue(box[0]<=150<box[2] and box[1]<=37.5<box[3])
        image=np.full((10,20,3),55,np.uint8);copy=image.copy();padded=square_rgb(image,28)
        self.assertEqual(padded.shape,(28,28,3));np.testing.assert_array_equal(image,copy)

    def test_rgb_missing_or_tied_descriptors_cannot_claim_distinctive_matches(self):
        blank=rgb_descriptors(np.full((64,64,3),100,np.uint8));features,score=rgb_correspondence_features(blank,blank,.01,.8)
        self.assertTrue(np.isfinite(features).all());self.assertEqual(score,0.)
        xy=np.array([[0,0],[1,0],[0,1],[1,1]],np.float32)
        features,score=correspondence_features(np.ones((4,4),np.float32),.01,xy,xy,.36)
        self.assertEqual(features[3],0.);self.assertEqual(score,0.)

    def test_non_grid_rgb_points_use_their_actual_image_coordinates(self):
        points=np.array([[.1,.1],[.8,.1],[.1,.8],[.8,.8],[.3,.4]],np.float32)
        destination=points*.6+[.1,.2];cv2.setRNGSeed(17)
        features,score=correspondence_features(np.eye(5,dtype=np.float32),.01,destination,points,.36)
        self.assertAlmostEqual(features[8],1.);self.assertGreater(score,0.)

    def test_single_and_partial_batches_follow_the_configured_encoder_arithmetic(self):
        class Encoder:
            def forward_features(self,x):return {'x_norm_patchtokens':torch.full((len(x),1,2),float(len(x)))}
        verifier=TemplateVerifier.__new__(TemplateVerifier);verifier.encoder=Encoder();verifier.device=torch.device('cpu')
        verifier.image_mean=torch.zeros(1,3,1,1);verifier.image_std=torch.ones(1,3,1,1);verifier.cfg={'feature_batch':4,'image_size':14}
        image=np.zeros((14,14,3),np.uint8)
        self.assertEqual(verifier.encode([image]).shape,(1,1,2));self.assertTrue(torch.all(verifier.encode([image]*5)==4))
        verifier.cfg['feature_batch']=2;self.assertTrue(torch.all(verifier.encode([image]*5)==2))


if __name__=='__main__':unittest.main()
