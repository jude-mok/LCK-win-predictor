import unittest

import numpy as np
import pandas as pd
import torch

from backend.training.player_history_ablation import train as original_train
from backend.training.player_subset_search import ALL, coarse_candidates, sample_weights, scaler_mask, train


class SubsetSearchTests(unittest.TestCase):
    def test_search_includes_baseline_singletons_and_single_removals(self):
        candidates = coarse_candidates()
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertIn(tuple(ALL), candidates)
        for i in ALL:
            self.assertIn((i,), candidates)
            self.assertIn(tuple(j for j in ALL if j != i), candidates)
        self.assertTrue(all(c and set(c).issubset(ALL) for c in candidates))

    def test_recent_weights_and_scaler_do_not_admit_future(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2024-01-01', '2025-01-01'])})
        np.testing.assert_array_equal(sample_weights(frame, True).numpy(), [.25, .5, 1.])
        np.testing.assert_array_equal(scaler_mask(frame, '2025_only'), [False, False, True])
        future = pd.DataFrame({'date': pd.to_datetime(['2026-01-01'])})
        with self.assertRaises(AssertionError):
            scaler_mask(future, 'pooled')
        with self.assertRaises(AssertionError):
            sample_weights(future, True)

    def test_equal_weight_training_reproduces_original(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        x = torch.randn(16, 2, 5, 13)
        y = torch.arange(16).remainder(2).float()
        old, _ = original_train(42, ALL, x, y, epochs=25)
        new, _, _ = train(42, ALL, x, y, torch.ones(16), epochs=25)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[name], value, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
