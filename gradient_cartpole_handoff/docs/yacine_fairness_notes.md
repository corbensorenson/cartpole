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

This packet defaults to continuous force and near-upright stabilization. If the public target starts collapsed/hanging below the cart and requires swing-up, this repo has not beaten that target. Add that initialization mode and retrain before making a claim.

Suggested wording if this succeeds before exact matching:

> We produced a uniform 6-link MuJoCo cart-pole near-upright stabilization baseline with checkpoint, evaluation JSON, and reset-free video evidence. This is not a swing-up result. Next step is to match Yacine's exact action/reward/initialization spec.
