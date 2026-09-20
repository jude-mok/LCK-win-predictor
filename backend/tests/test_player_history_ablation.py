import unittest

import numpy as np
import pandas as pd
import torch

from backend.training.player_history_ablation import combine, make_net


class HistoricalAblationTests(unittest.TestCase):
    def test_removal_preserves_retained_initial_weights_and_team_symmetry(self):
        full = make_net(42, list(range(13)))
        kept = [i for i in range(13) if i != 3]
        reduced = make_net(42, kept)
        torch.testing.assert_close(reduced.player[0].weight, full.player[0].weight[:, kept])
        for name, parameter in reduced.state_dict().items():
            if name != 'player.0.weight':
                torch.testing.assert_close(parameter, full.state_dict()[name])
        x = torch.randn(6, 2, 5, 12)
        torch.testing.assert_close(reduced(x), -reduced(x.flip(1)))
        self.assertEqual(sum(p.numel() for p in reduced.parameters()), 205)

    def test_historical_append_keeps_inputs_and_remaps_first_sets(self):
        base = pd.DataFrame({'gameid': ['a', 'b', 'c'],
                             'date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-02-01']),
                             'result': [1, 0, 1]})
        x = np.arange(3 * 2 * 5 * 13).reshape(3, 2, 5, 13)
        past = pd.DataFrame({'gameid': ['old'], 'date': pd.to_datetime(['2024-03-01']), 'result': [1]})
        series = pd.DataFrame({'date': base.date, 'game_index': [0, 1, 2]})
        frame, combined, ss = combine(base, x, series, {2024: (past, x[:1] * 0)},
                                      [2024, 2025], pd.Timestamp('2025-02-01'))
        self.assertEqual(frame.gameid.tolist(), ['old', 'a', 'b'])
        np.testing.assert_array_equal(combined[ss.game_index], x[:2])
        self.assertTrue(frame.date.lt('2025-02-01').all())


if __name__ == '__main__':
    unittest.main()
