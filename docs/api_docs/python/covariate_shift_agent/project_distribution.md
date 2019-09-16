<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="rainbow_agent.project_distribution" />
<meta itemprop="path" content="Stable" />
</div>

# covariate_shift_agent.project_distribution

```python
covariate_shift_agent.project_distribution(
    supports,
    weights,
    target_support,
    validate_args=False
)
```

Projects a batch of (support, weights) onto target_support. It is able to deal with non-equally spaced supports (e.g. with exponential ones).


#### Args:

*   <b>`supports`</b>: Tensor of shape (batch_size, num_dims) defining supports
    for the distribution.
*   <b>`weights`</b>: Tensor of shape (batch_size, num_dims) defining weights on
    the original support points. Although for the CategoricalDQN agent these
    weights are probabilities, it is not required that they are.
*   <b>`target_support`</b>: Tensor of shape (num_dims) defining support of the
    projected distribution. The values must be monotonically increasing. Vmin
    and Vmax will be inferred from the first and last elements of this tensor,
    respectively. The values in this tensor must be equally spaced.
*   <b>`validate_args`</b>: Whether we will verify the contents of the
    target_support parameter.

#### Returns:

A Tensor of shape (batch_size, num_dims) with the projection of a batch of
(support, weights) onto target_support.

#### Raises:

*   <b>`ValueError`</b>: If target_support has no dimensions, or if shapes of
    supports, weights, and target_support are incompatible.
