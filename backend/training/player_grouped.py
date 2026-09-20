"""Compare additive, joint and upper/lower player models under the same protocol."""
from backend.training.player_two_track import run,ROOT

if __name__=='__main__':
    run(tracks=['position_additive','cross_position','upper_lower'],
        output_dir=ROOT/'notebook/experiments/10_player_grouped')
