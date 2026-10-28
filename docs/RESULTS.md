# Evidence boundary

Final archived system, three seeds, observed-only exploratory test comparison:

| Method | Mean absolute error per 30-minute bin |
|---|---:|
| VORAFuel | 0.057894 |
| Random Forest | 0.059625 |
| Chronos-2, fuel-only | 0.138136 |
| Chronos-2, covariates | 0.138335 |
| TimesFM 2.5 | 0.094602 |
| Persistence | 0.106447 |

The manuscript uses provisional MT units pending provider confirmation of meter
unit, scale and measured equipment scope. There are 2,506 valid output bins per
seed, 210 valid first bins, and 34 complete six-hour windows in this comparison.
This is a single-vessel retrospective development evaluation, not an independent
unseen-vessel benchmark. The comparison includes task-specific fitting and input
differences; it is not an isolated architecture-effect estimate.

Day-block confidence intervals do not establish a VORAFuel–RF advantage. RF is
better on departure and non-overlapping evaluation, and some total-consumption
comparisons favor persistence or TimesFM. Do not infer fuel savings or operational
safety from these forecast errors.

Separate randomly initialized matched-budget supervised controls use 4,096
training windows, 640 updates and three seeds. Their full model MAE is 0.066983,
versus 0.069762 for GRU memory and 0.071581 for temporal Transformer memory.
Those controls are not the final multistage checkpoint and do not outperform RF.
Availability effects vary with period and condition.

The archived raw-to-selected-training replay had zero maximum final parameter and
prediction differences for seeds 13, 37 and 71. Packaging tests do not constitute
new research experiments or new validation evidence.
