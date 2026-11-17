"""Fixed-budget rank fusion, duplicate suppression and provenance."""
from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation.eval_proposal_fusion import fuse_candidates


def candidate(x, score=1.):
    return {'point_xy': [x / 100, .5], 'box_xyxy': [x, 0, x + 10, 10], 'logit': score}


class ProposalFusionTests(unittest.TestCase):
    def test_rank_schedule_ignores_incomparable_model_scores_and_preserves_inputs(self):
        pools = {'original': [candidate(0, -100), candidate(20, -200)],
                 'all_balanced': [candidate(40, 1000), candidate(60, 900)]}
        previous = copy.deepcopy(pools)
        selected, _ = fuse_candidates(pools, 'alternate_original', 3, .5)
        self.assertEqual([(r['source'], r['source_rank']) for r in selected],
                         [('original', 1), ('all_balanced', 1), ('original', 2)])
        self.assertEqual(pools, previous)

    def test_cross_source_duplicates_do_not_consume_a_budget_slot(self):
        pools = {'original': [candidate(0), candidate(20), candidate(40)],
                 'all_balanced': [candidate(1), candidate(60)]}
        selected, suppressed = fuse_candidates(pools, 'alternate_original', 3, .5)
        self.assertEqual(len(selected), 3)
        self.assertEqual([(r['source'], r['source_rank']) for r in selected],
                         [('original', 1), ('original', 2), ('all_balanced', 2)])
        self.assertEqual(suppressed[0]['source'], 'all_balanced')
        self.assertEqual(suppressed[0]['blocked_by_source'], 'original')

    def test_anchor_and_reverse_schedules_keep_their_declared_first_source(self):
        pools = {'original': [candidate(0), candidate(20)],
                 'all_balanced': [candidate(40), candidate(60)]}
        anchor, _ = fuse_candidates(pools, 'original_anchor', 3, .5)
        reverse, _ = fuse_candidates(pools, 'alternate_balanced', 1, .5)
        self.assertEqual([r['source'] for r in anchor], ['original', 'all_balanced', 'all_balanced'])
        self.assertEqual(reverse[0]['source'], 'all_balanced')

    def test_iou_equal_to_threshold_is_retained_and_exhausted_cache_is_not_padded(self):
        a = {'point_xy': [0, 0], 'box_xyxy': [0, 0, 3, 1], 'logit': 1.}
        b = {'point_xy': [1, 0], 'box_xyxy': [1, 0, 4, 1], 'logit': 1.}
        selected, suppressed = fuse_candidates({'original': [a], 'all_balanced': [b]}, 'alternate_original', 5, .5)
        self.assertEqual(len(selected), 2); self.assertEqual(suppressed, [])

    def test_single_source_points_boxes_and_rank_order_reproduce_source_prefix(self):
        pools = {'original': [candidate(0), candidate(20)], 'all_balanced': []}
        selected, _ = fuse_candidates(pools, 'original', 1, .5)
        self.assertEqual({k: selected[0][k] for k in pools['original'][0]}, pools['original'][0])
        self.assertEqual(selected[0]['source_rank'], 1)

    def test_zero_budget_and_empty_caches_return_no_candidates(self):
        pools = {'original': [candidate(0)], 'all_balanced': []}
        self.assertEqual(fuse_candidates(pools, 'alternate_original', 0, .5), ([], []))
        self.assertEqual(fuse_candidates({'original': [], 'all_balanced': []}, 'original_anchor', 5, .5), ([], []))


if __name__ == '__main__':
    unittest.main()
