import unittest
import numpy as np
import torch
from backend.training.player_two_track import PlayerNetwork,Preprocessor,INPUTS,series_probability

class PlayerTracksTests(unittest.TestCase):
    def test_series_probability(self):
        np.testing.assert_allclose(series_probability(np.array([.6,.6]),[3,5]),[.648,.68256])
        p=np.linspace(0,1,101)
        for bo in [3,5]:
            q=series_probability(p,bo)
            np.testing.assert_allclose(q+series_probability(1-p,bo),1,atol=1e-12)
            np.testing.assert_array_equal(q>=.5,p>=.5)

    def test_swapping_teams_reverses_logits(self):
        torch.manual_seed(42)
        x=torch.randn(7,2,5,len(INPUTS))
        for track in ['position_additive','cross_position','upper_lower']:
            net=PlayerNetwork(track)
            torch.testing.assert_close(net(x),-net(x.flip(1)))

    def test_additive_track_has_no_cross_role_interaction(self):
        torch.manual_seed(42)
        net=PlayerNetwork('position_additive');x=torch.randn(7,2,5,len(INPUTS))
        a=x.clone();a[:,0,0]+=2
        b=x.clone();b[:,0,1]-=3
        both=a.clone();both[:,0,1]-=3
        torch.testing.assert_close(net(both)-net(a)-net(b)+net(x),torch.zeros(7),atol=1e-6,rtol=0)

    def test_upper_lower_role_routing(self):
        torch.manual_seed(42)
        net=PlayerNetwork('upper_lower')
        x=torch.randn(3,2,5,len(INPUTS))
        captured={}
        def capture(name):
            def hook(module,args):
                captured[name]=args[0].detach().clone()
            return hook
        upper=net.upper.register_forward_pre_hook(capture('upper'))
        lower=net.lower.register_forward_pre_hook(capture('lower'))
        net(x)
        z=net.player(x)
        torch.testing.assert_close(captured['upper'],z[:,:,[0,1,2],:].flatten(start_dim=2))
        torch.testing.assert_close(captured['lower'],z[:,:,[3,4,1],:].flatten(start_dim=2))
        torch.testing.assert_close(captured['upper'][:,:,4:8],captured['lower'][:,:,8:12])
        upper.remove();lower.remove()
        self.assertEqual(sum(p.numel() for p in net.parameters()),225)

    def test_cold_start_transform_does_not_refit(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(8,2,5,len(INPUTS)))
        prep=Preprocessor().fit(x);mean=prep.scaler.mean_.copy()
        future=np.full((3,2,5,len(INPUTS)),np.nan)
        self.assertTrue(torch.isfinite(prep.transform(future)).all())
        np.testing.assert_array_equal(mean,prep.scaler.mean_)

if __name__=='__main__':unittest.main()
