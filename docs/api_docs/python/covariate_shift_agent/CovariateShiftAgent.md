<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="covariate_shift_agent.CovariateShiftAgent" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="begin_episode"/>
<meta itemprop="property" content="bundle_and_checkpoint"/>
<meta itemprop="property" content="end_episode"/>
<meta itemprop="property" content="step"/>
<meta itemprop="property" content="unbundle"/>
</div>

# covariate_shift_agent.CovariateShiftAgent

## Class `CovariateShiftAgent`

Inherits From: [`RainbowAgent`](../rainbow_agent/RainbowAgent.md)

Extension of a Rainbow agent in which we can predict and train covariate shift ratio estimates, as well as use them as priorities in an Off-Policy Setting.

<h2 id="__init__"><code>__init__</code></h2>

```python
__init__(
    *args,
    **kwargs
)
```

Initializes the agent and constructs the components of its graph.

#### Args:

*   <b>`sess`</b>: `tf.Session`, for executing ops.
*   <b>`num_actions`</b>: int, number of actions the agent can take at any
    state.
*   <b>`observation_shape`</b>: tuple of ints or an int. If single int, the
    observation is assumed to be a 2D square.
*   <b>`observation_dtype`</b>: tf.DType, specifies the type of the
    observations. Note that if your inputs are continuous, you should set this
    to tf.float32.
*   <b>`stack_size`</b>: int, number of frames to use in state stack.
*   <b>`network`</b>: function expecting three parameters: (num_actions,
    network_type, state). This function will return the network_type object
    containing the tensors output by the network. See
    dopamine.discrete_domains.atari_lib.rainbow_network as an example.
*   <b>`num_atoms`</b>: int, the number of buckets of the value function
    distribution.
*   <b>`vmax`</b>: float, the value distribution support is [-vmax, vmax].
*   <b>`gamma`</b>: float, discount factor with the usual RL meaning.
*   <b>`update_horizon`</b>: int, horizon at which updates are performed, the
    'n' in n-step update.
*   <b>`min_replay_history`</b>: int, number of transitions that should be
    experienced before the agent begins training its value function.
*   <b>`update_period`</b>: int, period between DQN updates.
*   <b>`target_update_period`</b>: int, update period for the target network.
*   <b>`epsilon_fn`</b>: function expecting 4 parameters: (decay_period, step,
    warmup_steps, epsilon). This function should return the epsilon value used
    for exploration during training.
*   <b>`epsilon_train`</b>: float, the value to which the agent's epsilon is
    eventually decayed during training.
*   <b>`epsilon_eval`</b>: float, epsilon used when evaluating the agent.
*   <b>`epsilon_decay_period`</b>: int, length of the epsilon decay schedule.
*   <b>`replay_scheme`</b>: str, 'prioritized' or 'uniform', the sampling scheme
    of the replay memory.
*   <b>`tf_device`</b>: str, Tensorflow device on which the agent's graph is
    executed.
*   <b>`use_staging`</b>: bool, when True use a staging area to prefetch the
    next training batch, speeding training up by about 30%.
*   <b>`optimizer`</b>: `tf.train.Optimizer`, for training the value function.
*   <b>`summary_writer`</b>: SummaryWriter object for outputting training
    statistics. Summary writing disabled if set to None.
*   <b>`summary_writing_frequency`</b>: int, frequency with which summaries will be written. Lower values will result in slower training.
*   <b>`use_ratio_model`</b>: bool, whether to train the ratio model or not,
*   <b>`use_priorities`</b>: bool, whether to use priorities from the ratio model
*   <b>`update_beginning_priorities`</b>: bool, whether to set beginning states to c=1
*   <b>`quotient_epsilon`</b>: float, epsilon used when computing the quotient of policies
*   <b>`quotient_epsilon_decay_period`</b>: int, length of the quotient epsilon decay schedule.
*   <b>`use_loss_weights`</b>: bool, whether to use loss weights of the Q model
*   <b>`ratio_num_atoms`</b>: int, the number of buckets of the ratio function distribution.
*   <b>`ratio_cmin`</b>: float, the predefined minimum ratio value
*   <b>`ratio_cmax`</b>: float, the predefined maximum ratio value
*   <b>`symmetric_range`</b>: bool, whether to define a symmetric range with respect to the ratio
        updates,
*   <b>`symmetric_exponent`</b>: int, number of update multiplications that are contained in
        the range,
*   <b>`log_ratio_approach`</b>: bool, whether to consider the logarithmic update approach
*   <b>`use_ratio_exp_bins`</b>: bool, whether to use an exponential sequence of bins 
        instead of linear ones
*   <b>`plot_log_scale`</b>: bool, whether to show x-axis distribution plot in log scale when
        using exponential bins
*   <b>`define_base_and_exp`</b>: bool, whether the base and exponents are defined; otherwise,
        the default interval is divided in an exponential way
*   <b>`ratio_exp_base`</b>: float, base of the exponential sequence
*   <b>`ratio_min_exp`</b>: int, minimum exponent
*   <b>`ratio_max_exp`</b>: int, maximum exponent
*   <b>`ratio_discount_factor`</b>: float, discount factor used in Discounted COP-TD
*   <b>`ratio_loss_weight`</b>: float, loss weight of the covariate shift ratio estimation

## Methods

<h3 id="begin_episode"><code>begin_episode</code></h3>

```python
begin_episode(observation)
```

Returns the agent's first action for this episode, and sets a flag to indicate that we are at the very beginning of the episode.

#### Args:

*   <b>`observation`</b>: numpy array, the environment's initial observation.

#### Returns:

int, the selected action.

<h3 id="bundle_and_checkpoint"><code>bundle_and_checkpoint</code></h3>

```python
bundle_and_checkpoint(
    checkpoint_dir,
    iteration_number
)
```

Returns a self-contained bundle of the agent's state.

This is used for checkpointing. It will return a dictionary containing all
non-TensorFlow objects (to be saved into a file by the caller), and it saves all
TensorFlow objects into a checkpoint file.

#### Args:

*   <b>`checkpoint_dir`</b>: str, directory where TensorFlow objects will be
    saved.
*   <b>`iteration_number`</b>: int, iteration number to use for naming the
    checkpoint file.

#### Returns:

A dict containing additional Python objects to be checkpointed by the
experiment. If the checkpoint directory does not exist, returns None.

<h3 id="end_episode"><code>end_episode</code></h3>

```python
end_episode(reward)
```

Signals the end of the episode to the agent.

We store the observation of the current time step, which is the last observation
of the episode.

#### Args:

*   <b>`reward`</b>: float, the last reward from the environment.

<h3 id="step"><code>step</code></h3>

```python
step(
    reward,
    observation
)
```

Records the most recent transition and returns the agent's next action.

We store the observation of the last time step since we want to store it with
the reward.

#### Args:

*   <b>`reward`</b>: float, the reward received from the agent's most recent
    action.
*   <b>`observation`</b>: numpy array, the most recent observation.

#### Returns:

int, the selected action.

<h3 id="unbundle"><code>unbundle</code></h3>

```python
unbundle(
    checkpoint_dir,
    iteration_number,
    bundle_dictionary
)
```

Restores the agent from a checkpoint.

Restores the agent's Python objects to those specified in bundle_dictionary, and
restores the TensorFlow objects to those specified in the checkpoint_dir. If the
checkpoint_dir does not exist, will not reset the agent's state.

#### Args:

*   <b>`checkpoint_dir`</b>: str, path to the checkpoint saved by tf.Save.
*   <b>`iteration_number`</b>: int, checkpoint version, used when restoring
    replay buffer.
*   <b>`bundle_dictionary`</b>: dict, containing additional Python objects owned
    by the agent.

#### Returns:

bool, True if unbundling was successful.
