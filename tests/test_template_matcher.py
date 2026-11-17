"""Template conditioning, unknown-pixel loss and target-independent confidence calibration."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import tempfile
import json
import h5py
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from perception.template_matcher import TemplateMatcher,localization_loss,template_pair_loss,presence_score
from training.train_template_matcher import calibrate,pair_plan,metrics,confidence_fold_plan,selection_key,load_features,response_scores,supervision
from training.extract_template_features import digest


class TemplateMatcherTests(unittest.TestCase):
    def test_template_changes_prediction_and_receives_gradients(self):
        torch.manual_seed(17);torch.set_num_threads(2)
        model=TemplateMatcher(feature_dim=16,hidden_dim=8,output_grid=8)
        search=torch.randn(2,16,16);template=torch.randn(2,16,16,requires_grad=True)
        logits,features=model(search,template)
        changed,_=model(search,-template)
        self.assertEqual(logits.shape,(2,8,8));self.assertEqual(features.shape,(2,10))
        self.assertGreater(float((changed-logits).abs().max().detach()),1e-5)
        logits.square().mean().backward();self.assertGreater(float(template.grad.abs().sum().detach()),0.)

    def test_search_only_control_cannot_depend_on_template(self):
        torch.manual_seed(17)
        model=TemplateMatcher(feature_dim=16,hidden_dim=8,output_grid=8,conditioned=False)
        query=torch.randn(1,16,16);a,_=model(query,torch.randn(1,16,16));b,_=model(query,torch.randn(1,16,16))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_unknown_pixels_have_zero_loss_gradient(self):
        logits=torch.randn(1,4,4,requires_grad=True);target=torch.zeros(1,4,4);target[0,1,1]=1
        valid=torch.ones(1,4,4);valid[:,:,3]=0
        loss=localization_loss(logits,target,valid);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(float(logits.grad[:,:,3].abs().sum()),0.)
        all_unknown=localization_loss(logits,target,torch.zeros_like(valid))
        self.assertEqual(float(all_unknown),0.)

    def test_calibration_cannot_split_tied_scores_to_claim_precision(self):
        threshold,result=calibrate([.9,.9,.5],[True,False,True],.95)
        self.assertGreater(threshold,1);self.assertEqual(result['accepted'],0)
        threshold,result=calibrate([.9,.8,.7],[True,True,False],.95)
        self.assertTrue(.7<threshold<.8);self.assertEqual(result['accepted'],2)
        np.testing.assert_array_equal(np.array([.9,.8-2e-7,.7+2e-7])>=threshold,[True,True,False])
        high=.8;low=np.nextafter(high,0.)
        threshold,result=calibrate([high,low],[True,False],.95)
        self.assertGreater(threshold,low);self.assertLessEqual(threshold,high)

    def test_negative_template_group_is_rejected_if_any_copy_is_visible(self):
        def item(name,group,bounds):return {'id':name,'appearance_group':group,'category':'aircraft','bounds_ned_m':bounds}
        front=[[8,-1,-1],[10,1,1]];back=[[-10,-1,-1],[-8,1,1]]
        targets=[item('a','a',front),item('b','b',back),item('b-copy','b',front),item('c','c',back)]
        record={'target_id':'a','horizontal_fov_deg':90,'camera_metadata':dict(width=896,height=504,pos_x=0,pos_y=0,pos_z=0,rot_x=0,rot_y=0,rot_z=0,rot_w=1)}
        data={'catalog':{'targets':targets,'splits':{'appearance':{'train':[t['id'] for t in targets],'validation':[],'test':[]}}},
              'split':{'train':[0],'validation':[],'test':[]},'records':[record],'targets':[t['id'] for t in targets],
              'groups':{t['id']:t for t in targets}}
        plan=pair_plan(data,{'protocol':'appearance','seed':7,'negative_visibility_margin_deg':2})
        self.assertEqual(plan['train'][1]['negative_template_id'],'c')
        data['training_template_ids']=['a','b']
        restricted=pair_plan(data,{'protocol':'appearance','seed':7,'negative_visibility_margin_deg':2})
        self.assertEqual(len(restricted['train']),1)

    def test_wrong_template_acceptance_counts_as_false_even_at_a_visible_object(self):
        mask=np.ones((8,8),np.uint8)
        data={'native':[mask,mask]}
        rows=[{'query':0,'positive':True,'target_id':'a'},{'query':1,'positive':False,'target_id':'a'}]
        result=metrics(data,rows,np.array([[.5,.5],[.5,.5]]),np.array([.9,.9]),.8)
        self.assertEqual(result['accepted_precision'],.5)
        self.assertEqual(result['negative_false_accept_rate'],1.)

    def test_pair_loss_separates_templates_without_unknown_pixel_gradients(self):
        positive=torch.zeros(1,4,4,requires_grad=True);negative=torch.zeros_like(positive,requires_grad=True)
        target=torch.zeros_like(positive);target[0,1,1]=1;target[0,1,3]=1
        valid=torch.ones_like(positive);valid[:,:,3]=0
        loss=template_pair_loss(positive,negative,target,valid);loss.backward()
        self.assertLess(float(positive.grad[0,1,1]),0);self.assertGreater(float(negative.grad[0,1,1]),0)
        self.assertEqual(float(positive.grad[:,:,3].abs().sum()),0.)
        better=template_pair_loss(positive.detach()+3,negative.detach(),target,valid)
        self.assertLess(float(better),float(loss.detach()))
        self.assertEqual(float(template_pair_loss(positive,negative,target,torch.zeros_like(valid)).detach()),0.)

    def test_response_existence_score_handles_nondefault_output_grids(self):
        torch.manual_seed(3);model=TemplateMatcher(feature_dim=16,hidden_dim=8,output_grid=8)
        with torch.no_grad():logits,stats=model(torch.randn(2,16,16),torch.randn(2,16,16))
        np.testing.assert_allclose(response_scores(stats,8),presence_score(logits).numpy(),rtol=1e-6)

    def test_selection_prefers_verified_recall_and_identity_over_raw_hit(self):
        detector={'verified_positive_recall':0.,'verified_auc':.6,'macro_target_hit':.99}
        identity={'verified_positive_recall':.3,'verified_auc':.8,'macro_target_hit':.9}
        self.assertGreater(selection_key(identity),selection_key(detector))
        self.assertGreater(selection_key(detector,False),selection_key(identity,False))
        identity['verified_positive_recall']=0.
        self.assertGreater(selection_key(identity),selection_key(detector))

    def test_confidence_folds_keep_primary_mesh_copies_out_of_fitting(self):
        targets={'a':{'appearance_group':'g1'},'a-copy':{'appearance_group':'g1'},'b':{'appearance_group':'g2'},'c':{'appearance_group':'g3'}}
        records=[{'target_id':t} for t in targets for _ in range(2)]
        data={'groups':targets,'records':records,'split':{'train':list(range(len(records)))}}
        folds=confidence_fold_plan(data,3,17);seen=[]
        for fold in folds:
            self.assertFalse(set(fold['fit_queries'])&set(fold['held_queries']))
            seen.extend(fold['held_queries'])
            held={targets[records[i]['target_id']]['appearance_group'] for i in fold['held_queries']}
            self.assertTrue(all(targets[t]['appearance_group'] not in held for t in fold['fit_template_ids']))
        self.assertEqual(sorted(seen),list(range(len(records))))

    def test_development_loader_never_opens_test_shards_or_templates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);features=root/'features';features.mkdir()
            ids=['a','b','test'];catalog={'targets':[{'id':t} for t in ids],
                'splits':{'appearance':dict(train=['a'],validation=['b'],test=['test'])}}
            cat=root/'catalog.json';cat.write_text(json.dumps(catalog));entries=[]
            with h5py.File(features/'templates.h5','w') as f:
                for target in ids[:2]:
                    g=f.create_group(target);g.create_dataset('z',data=np.zeros((1,4,384),np.float16));g.attrs['variants_json']='[{}]'
            for target in ids[:2]:
                path=features/(target+'.h5')
                with h5py.File(path,'w') as f:
                    f.create_dataset('z',data=np.zeros((1,4,384),np.float16))
                    for key in ['mask64','valid64','native_mask']:f.create_dataset(key,data=np.ones((1,8,8),np.uint8))
                    f.create_dataset('metadata',data=[json.dumps({'target_id':target})],dtype=h5py.string_dtype('utf-8'))
                entries.append({'target_id':target,'file':path.name,'sha256':digest(path)})
            entries.append({'target_id':'test','file':'MUST_NOT_OPEN.h5','sha256':'unread'})
            index={'status':'partial','signature':{'catalog_sha256':digest(cat)},'template_file':'templates.h5',
                   'template_sha256':digest(features/'templates.h5'),'targets':entries}
            (features/'index.json').write_text(json.dumps(index))
            cfg={'feature_directory':str(features),'catalog':str(cat),'protocol':'appearance'}
            with mock.patch('h5py.File',wraps=h5py.File) as opened:data=load_features(cfg,'cpu',('train','validation'))
            self.assertEqual(data['targets'],['a','b']);self.assertNotIn('test',data['split'])
            self.assertTrue(all('MUST_NOT_OPEN' not in str(call.args[0]) for call in opened.call_args_list))

    def test_context_positive_uses_its_requested_target_mask(self):
        base=torch.zeros(1,4,4);base[0,0,0]=1;other=torch.zeros(4,4);other[3,3]=1
        data={'mask':base,'valid':torch.ones_like(base),'targets':['main','other'],
              'context_mask':{(0,'other'):other},'context_valid':{0:torch.ones(4,4)}}
        rows=[{'query':0,'template':0,'positive':True},{'query':0,'template':1,'positive':True,'context_positive':True}]
        masks,valid=supervision(data,rows)
        self.assertEqual(float(masks[0,0,0]),1.);self.assertEqual(float(masks[1,0,0]),0.)
        self.assertEqual(float(masks[1,3,3]),1.);self.assertEqual(float(data['mask'][0,0,0]),1.)


if __name__=='__main__':unittest.main()
