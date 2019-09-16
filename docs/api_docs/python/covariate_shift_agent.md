<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="covariate_shift_agent" />
<meta itemprop="path" content="Stable" />
</div>

# Module: covariate_shift_agent

Extension of a Rainbow agent in which we can predict and train covariate shift ratio estimates, as well as use them as priorities in an Off-Policy Setting.


## Classes

[`class CovariateShiftAgent`](./covariate_shift_agent/CovariateShiftAgent.md): An implementation of a simplified Rainbow agent that also deals with Covariate Shift Ratios.

## Functions

[`project_distribution(...)`](./covariate_shift_agent/project_distribution.md): Projects
a batch of (support, weights) onto target_support; in contrast to the [`project_distribution`](./rainbow_agent/project_distribution.md) function of Rainbow agent, this implementation allows supports that are non-equally spaced.