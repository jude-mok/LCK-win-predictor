# Preliminary numerical audit — not the authoritative experiment14 result

This preliminary run used an algebraically equivalent weighted-sum BCE even for
equal-weight examples. Its float32 backward path differed from the original
mean-reduction BCE. Small accumulated differences changed the baseline2025
chronological CV from79 to80 correct out of123, so this run was superseded before
the results were delivered or used as the model-selection conclusion.

The corrected implementation uses the original BCE mean path whenever all
weights are equal, checks exact parameter agreement in a unit test, and verifies
the original79/123 CV score before searching. The authoritative outputs are in
`../14_player_subset_search/`. This directory is retained only as an audit trail;
do not mix these scores, candidates, or weights with the corrected run.
