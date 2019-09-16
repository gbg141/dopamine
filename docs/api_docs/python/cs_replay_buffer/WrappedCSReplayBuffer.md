<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cs_replay_buffer.WrappedCSReplayBuffer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add"/>
<meta itemprop="property" content="create_sampling_ops"/>
<meta itemprop="property" content="load"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="tf_get_priority"/>
<meta itemprop="property" content="tf_set_priority"/>
<meta itemprop="property" content="unpack_transition"/>
</div>

# cs_replay_buffer.WrappedCSReplayBuffer

## Class `WrappedCSReplayBuffer`

Inherits From:
[`WrappedPrioritizedReplayBuffer`](../prioritized_replay_buffer/WrappedPrioritizedReplayBuffer.md)

Wrapper of OutOfGraphPrioritizedReplayBuffer, with both uniform and in-graph sampling, that also manages the storage and sampling of Covariate Shift Ratios.

Usage:

*   To add a transition: Call the add function.

*   To sample a batch: Query any of the tensors in the transition dictionary.
    Every sess.run that requires any of these tensors will sample a new
    transition.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    *args,
    **kwargs
)
```

Initializes WrappedPrioritizedReplayBuffer.

#### Args:

*   <b>`observation_shape`</b>: tuple of ints.
*   <b>`stack_size`</b>: int, number of frames to use in state stack.
*   <b>`use_staging`</b>: bool, when True it would use a staging area to
    prefetch the next sampling batch.
*   <b>`replay_capacity`</b>: int, number of transitions to keep in memory.
*   <b>`batch_size`</b>: int.
*   <b>`update_horizon`</b>: int, length of update ('n' in n-step update).
*   <b>`gamma`</b>: int, the discount factor.
*   <b>`max_sample_attempts`</b>: int, the maximum number of attempts allowed to
    get a sample.
*   <b>`extra_storage_types`</b>: list of ReplayElements defining the type of
    the extra contents that will be stored and returned by
    sample_transition_batch.
*   <b>`observation_dtype`</b>: np.dtype, type of the observations. Defaults to
    np.uint8 for Atari 2600.
*   <b>`action_shape`</b>: tuple of ints, the shape for the action vector. Empty
    tuple means the action is a scalar.
*   <b>`action_dtype`</b>: np.dtype, type of elements in the action.
*   <b>`reward_shape`</b>: tuple of ints, the shape of the reward vector. Empty
    tuple means the reward is a scalar.
*   <b>`reward_dtype`</b>: np.dtype, type of elements in the reward.

#### Raises:

*   <b>`ValueError`</b>: If update_horizon is not positive.
*   <b>`ValueError`</b>: If discount factor is not in [0, 1].

## Methods

<h3 id="create_sampling_ops"><code>create_sampling_ops</code></h3>

```python
create_sampling_ops(
    use_staging
)
```

Creates the ops necessary to sample from the replay buffer.

Creates the transition dictionaries, each containing one of the types of sampling tensors:
* transition_tensors: sampled according to priorities for learning q values
* 'uniform_transition_tensors': uniformly sampled for learning covariate shift ratio estimates.

#### Args:

*   <b>`use_staging`</b>: bool, when True it would use a staging area to
    prefetch the next sampling batch.

<h3 id="load"><code>load</code></h3>

```python
load(
    checkpoint_dir,
    suffix
)
```

Loads the replay buffer's state from a saved file.

#### Args:

*   <b>`checkpoint_dir`</b>: str, the directory where to read the numpy
    checkpointed files from.
*   <b>`suffix`</b>: str, the suffix to use in numpy checkpoint files.

<h3 id="save"><code>save</code></h3>

```python
save(
    checkpoint_dir,
    iteration_number
)
```

Save the underlying replay buffer's contents in a file.

#### Args:

*   <b>`checkpoint_dir`</b>: str, the directory where to read the numpy
    checkpointed files from.
*   <b>`iteration_number`</b>: int, the iteration_number to use as a suffix in
    naming numpy checkpoint files.

<h3 id="tf_get_priority"><code>tf_get_priority</code></h3>

```python
tf_get_priority(indices)
```

Gets the priorities for the given indices.

#### Args:

*   <b>`indices`</b>: tf.Tensor with dtype int32 and shape [n].

#### Returns:

*   <b>`priorities`</b>: tf.Tensor with dtype float and shape [n], the
    priorities at the indices.

<h3 id="tf_set_priority"><code>tf_set_priority</code></h3>

```python
tf_set_priority(
    indices,
    priorities
)
```

Sets the priorities for the given indices.

#### Args:

*   <b>`indices`</b>: tf.Tensor with dtype int32 and shape [n].
*   <b>`priorities`</b>: tf.Tensor with dtype float and shape [n].

#### Returns:

A tf op setting the priorities for prioritized sampling.

<h3 id="unpack_transition"><code>unpack_transition</code></h3>

```python
unpack_transition(
    transition_tensors,
    uniform_transition_tensors,
    transition_type
)
```

Unpacks the given transitions (prioritized and uniform) into member variables.

#### Args:

*   <b>`transition_tensors`</b>: tuple of tf.Tensors.
*   <b>`uniform_transition_tensors`</b>: tuple of tf.Tensors uniformly sampled.
*   <b>`transition_type`</b>: tuple of ReplayElements matching
    transition_tensors.
