# Fairness Notes for Comparing Against Yacine's Benchmark

Before claiming a direct beat, write down the exact target environment:

- Number of links.
- Link lengths.
- Link masses.
- Joint damping/friction.
- Cart mass.
- Rail length.
- Force limit.
- Continuous or discrete action space.
- If discrete: exact force bins.
- Initial state distribution.
- Reward function.
- Episode length.
- Termination thresholds.
- MuJoCo version.
- Policy observation vector.

This packet defaults to continuous force and near-upright stabilization. If the public target uses five discrete force actions or swing-up from non-upright states, add those modes and retrain before making a claim.

Suggested wording if this succeeds before exact matching:

> We trained a 6-link MuJoCo cart-pole to stabilize from near-upright using a gradient-to-uniform curriculum. The morphology is uniform at final evaluation. Next step is to match Yacine's exact action/reward/initialization spec.

