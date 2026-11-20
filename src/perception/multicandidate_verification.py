"""Frozen image-only candidate fusion and template verification, also used in shadow mode."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from common.paths import project_path
from perception.extract_dino_features import to_input_tensor
from perception.template_verification import (TemplateVerifier, square_rgb, rgb_descriptors,
                                              rgb_correspondence_features, patch_verification_features)
from perception.template_matcher import TemplateMatcher
from evaluation.eval_template_selection import peak_candidates
from evaluation.eval_proposal_fusion import fuse_candidates


class MultiCandidateVerifier(TemplateVerifier):
    def __init__(self, configuration, device='cuda', exported=None):
        cfg = configuration
        super().__init__(cfg['base_verifier'], device)
        self.protocol = cfg; self.reference_bank = {}; self.heads = {}; self.states = {}
        self._sift_pool = ThreadPoolExecutor(max_workers=max(1, min(8, int(cfg.get('candidate_count', 5)))))
        for fold in range(3):
            for source in ['original', 'all_balanced']:
                path = project_path(cfg['proposal_directory']) / source / f'fold-{fold}/model.pt'
                state = torch.load(path, map_location='cpu', weights_only=True)
                params = state['configuration']
                model = TemplateMatcher(conditioned=False, hidden_dim=params['hidden_dim'], output_grid=params['output_grid']).to(self.device)
                model.load_state_dict(state['model']); model.eval()
                self.heads[fold, source] = model; self.states[fold, source] = state
        self.acceptance = None
        if exported is not None:
            self.acceptance = torch.load(project_path(exported), map_location='cpu', weights_only=True)

    @torch.no_grad()
    def _encode_fast(self, images):
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            x = to_input_tensor(np.stack(images), self.cfg['image_size'], self.image_mean, self.image_std, self.device)
            return self.encoder.forward_features(x)['x_norm_patchtokens'].to(torch.float16).float()

    def add_reference(self, name, rgb):
        self.set_reference(rgb)
        self.reference_bank[name] = (self.reference, self.rgb_reference)

    @torch.no_grad()
    def prepare_query(self, rgb, fold=None):
        fold = self.protocol['deployment_fold'] if fold is None else fold
        tokens = self._encode_fast([rgb]); pools = {}; statistics = {}
        for source in ['original', 'all_balanced']:
            state = self.states[fold, source]
            q = (tokens - state['mean'].to(self.device)) / state['std'].to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                logits, stats = self.heads[fold, source](q, torch.zeros_like(q))
            statistics[source] = stats[0].cpu().numpy()
            pools[source] = peak_candidates(logits[0].cpu().numpy(), rgb.shape[:2], self.cfg,
                                            self.protocol['candidate_count'], self.protocol['nms_iou'])
        candidates, _ = fuse_candidates(pools, self.protocol['fusion'], self.protocol['candidate_count'], self.protocol['nms_iou'])
        crops = [square_rgb(rgb[y0:y1, x0:x1], self.cfg['image_size']) for x0, y0, x1, y1 in [c['box_xyxy'] for c in candidates]]
        if not crops:
            return dict(candidates=[], tokens=[], rgb=[], statistics=statistics)
        pool = getattr(self, '_sift_pool', None)
        sift_futures = [pool.submit(rgb_descriptors, im) for im in crops] if pool is not None else None
        z = self._encode_fast(crops)
        crop_tokens = F.normalize((z - self.crop_mean) / self.crop_std, dim=-1)
        crop_rgb = [f.result() for f in sift_futures] if sift_futures is not None else [rgb_descriptors(im) for im in crops]
        return dict(candidates=candidates, tokens=crop_tokens, rgb=crop_rgb, statistics=statistics)

    def candidate_features(self, query, reference_id):
        reference, rgb_reference = self.reference_bank[reference_id]; result = []
        for i, candidate in enumerate(query['candidates']):
            cv2.setRNGSeed(self.protocol['confidence_seed'])
            vec, _ = patch_verification_features(query['tokens'][i], reference,
                        query['statistics'][candidate['source']], self.cfg['reprojection_threshold'])
            rgb_vec, _ = rgb_correspondence_features(query['rgb'][i], rgb_reference,
                        self.cfg['rgb_reprojection_threshold'], self.cfg['rgb_ratio'], device=self.device)
            result.append(np.r_[vec, rgb_vec].astype(np.float32))
        return np.asarray(result, np.float32).reshape(-1, 44)

    def score_candidate(self, query, index, reference_id):
        if self.acceptance is None:
            raise RuntimeError('Load a frozen acceptance export before inference')
        if index < 0 or index >= len(query['candidates']):
            raise IndexError('Candidate index out of range')
        reference, rgb_reference = self.reference_bank[reference_id]
        candidate = query['candidates'][index]
        cv2.setRNGSeed(self.protocol['confidence_seed'])
        vec, _ = patch_verification_features(query['tokens'][index], reference,
                    query['statistics'][candidate['source']], self.cfg['reprojection_threshold'])
        rgb_vec, _ = rgb_correspondence_features(query['rgb'][index], rgb_reference,
                    self.cfg['rgb_reprojection_threshold'], self.cfg['rgb_ratio'], device=self.device)
        value = np.r_[vec, rgb_vec].astype(np.float32)
        state = self.acceptance
        logit = float(((value - state['center'].numpy()) / state['scale'].numpy()) @ state['weight'].numpy() + float(state['bias']))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -700.0, 700.0))))

    def select(self, query, reference_id):
        if self.acceptance is None:
            raise RuntimeError('Load a frozen acceptance export before inference')
        values = self.candidate_features(query, reference_id)
        state = self.acceptance
        if not len(values):
            return dict(accepted=False, point_xy=None, score=None, candidates=[], chosen_rank=None)
        logits = ((values - state['center'].numpy()) / state['scale'].numpy()) @ state['weight'].numpy() + float(state['bias'])
        scores = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -700, 700)))
        index = int(np.argmax(scores)); threshold = state['threshold']
        return dict(accepted=bool(threshold is not None and scores[index] >= threshold),
                    point_xy=query['candidates'][index]['point_xy'], score=float(scores[index]), chosen_rank=index+1,
                    candidates=[dict(c, verification_score=float(s)) for c, s in zip(query['candidates'], scores)])

    def predict(self, rgb, reference_id):
        return self.select(self.prepare_query(rgb), reference_id)
