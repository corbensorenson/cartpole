# Experiment Protocol

## Core hypothesis

For an n-link cart-pole, direct training on uniform links is unnecessarily hard. A gradiented morphology can improve practical controllability. A curriculum can then anneal gradients away until the plant is uniform.

## Morphology generator

For link index `i = 1..n`, define normalized position:

```math
s_i = \frac{i-1}{n-1}
```

For a parameter budget `P` and gradient strength `alpha_p`:

```math
p_i = P \frac{e^{-\alpha_p s_i}}{\sum_j e^{-\alpha_p s_j}}
```

This is used for:

```math
\ell_i, m_i, d_i
```

Positive `alpha` means base-heavy/base-long/base-damped. `alpha=0` is uniform.

## Curriculum

Default mode: `mass_last`.

1. Remove damping gradient first.
2. Remove damping budget toward target.
3. Remove length gradient.
4. Remove mass gradient last.

Reasoning: damping is the easiest/passive crutch, length gradients can create fast top modes, and mass gradient seems like the strongest practical conditioning knob.

## Required controls

Run these eventually:

| Run | Purpose |
|---|---|
| uniform direct PPO | Baseline difficulty |
| length-gradient only | Tests musical/length intuition |
| mass-gradient only | Tests strongest current hypothesis |
| full gradient | Tests combined curriculum |
| shuffled gradient | Tests whether ordered gradient matters |
| gradient-to-uniform | Actual claim |

## Acceptance metrics

For each run record:

- wall-clock time
- total environment steps
- mean return
- success rate
- mean episode length
- max absolute link angle
- checkpoint path
- video path
- config hash / config copy

