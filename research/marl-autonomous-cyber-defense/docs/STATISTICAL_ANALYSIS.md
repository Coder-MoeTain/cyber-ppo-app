# Statistical analysis plan

## Experimental unit and endpoints
The independent unit is a complete evaluation episode under a predeclared scenario/seed. Pair algorithms on matched scenario and seed configurations. Predeclare critical-asset compromise rate and service availability as primary endpoints. Secondary endpoints: attack success, mitigation time, unnecessary defensive actions, communication volume, inference latency and training compute.

## Analysis
- Report episode-level mean, standard deviation, median and interquartile range.
- Report paired mean differences and bootstrap confidence intervals.
- Use paired sign-flip randomization tests where assumptions hold.
- Apply Holm-Bonferroni correction across predeclared confirmatory comparisons.
- Report effect sizes, raw and adjusted p-values, and negative/null results.
- Do not treat timesteps, hosts or agents within an episode as independent samples.

The included utilities are analysis primitives, not a substitute for justifying the sampling design, test assumptions, multiplicity family and interval method.
