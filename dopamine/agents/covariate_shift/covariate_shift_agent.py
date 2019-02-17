# coding=utf-8
# gbg141

"""Agent permorming COP-TD

Details in "Off-Policy Deep Reinforcement Learning by Bootstrapping the Covariate Shift" by 
Carles Gelada & Marc G. Bellemare (2018)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.replay_memory import prioritized_replay_buffer, cs_replay_buffer
import numpy as np
import tensorflow as tf

import gin.tf

slim = tf.contrib.slim


@gin.configurable
class CovariateShiftAgent(rainbow_agent.RainbowAgent):
  """A compact implementation of a simplified Rainbow agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=1.0,
               epsilon_eval=0.1,
               epsilon_decay_period=250000,
               replay_scheme='uniform',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500,
               ratio_num_atoms=51,
               ratio_cmin=0.,
               ratio_cmax=10.,
               ratio_discount_factor=0.99,
               ratio_loss_weight=0.02):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      ratio_num_atoms: int, the number of buckets of the ratio function distribution.
      ratio_cmin: float, the predefined minimum ratio value
      ratio_cmax: float, the predefined maximum ratio value
      ratio_discount_factor: float, discount factor used in Discounted COP-TD
      ratio_loss_weight: float, loss weight of the covariate shift ratio estimation
    """
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    self._support = tf.linspace(-vmax, vmax, num_atoms)
    self._replay_scheme = replay_scheme
    self.optimizer = optimizer

    # Initializing extra parameters
    ratio_cmin = float(ratio_cmin)
    ratio_cmax = float(ratio_cmax)
    self._ratio_num_atoms = ratio_num_atoms
    self._ratio_support = tf.linspace(ratio_cmin, ratio_cmax, ratio_num_atoms)
    self.ratio_discount_factor = ratio_discount_factor
    self.ratio_loss_weight = ratio_loss_weight

    super(CovariateShiftAgent, self).__init__(
          sess=sess,
          num_actions=num_actions,
          observation_shape=observation_shape,
          observation_dtype=observation_dtype,
          stack_size=stack_size,
          gamma=gamma,
          update_horizon=update_horizon,
          min_replay_history=min_replay_history,
          update_period=update_period,
          target_update_period=target_update_period,
          epsilon_fn=epsilon_fn,
          epsilon_train=epsilon_train,
          epsilon_eval=epsilon_eval,
          epsilon_decay_period=epsilon_decay_period,
          tf_device=tf_device,
          use_staging=use_staging,
          optimizer=self.optimizer,
          summary_writer=summary_writer,
          summary_writing_frequency=summary_writing_frequency)

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    action = super(CovariateShiftAgent, self).begin_episode(observation)
    self.is_beginning = True
    return action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False, self.is_beginning)
      self._train_step()

    self.action = self._select_action()
    self.is_beginning = False
    return self.action

  def _store_transition(self, last_observation, action, reward, is_terminal, is_beginning=False):
    """Stores an experienced transition.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer:
      (last_observation, action, reward, is_terminal).

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
      is_beginning: bool, indicating if the current state is a beginning state.
    """
    self._replay.add(last_observation, action, reward, is_terminal, is_beginning)

  def _get_network_type(self):
    """Returns the type of the outputs of a value distribution network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('cs_network',
                                  ['c_values', 'c_logits', 'c_probabilities',
                                   'q_values', 'logits', 'probabilities'])

  def _network_template(self, state):
    """Builds a convolutional network that outputs CS ratio and Q-value distributions.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = slim.conv2d(
        net, 32, [8, 8], stride=4, weights_initializer=weights_initializer)
    net = slim.conv2d(
        net, 64, [4, 4], stride=2, weights_initializer=weights_initializer)
    net = slim.conv2d(
        net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    net = slim.flatten(net)

    ratio_net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)
    ratio_net = slim.fully_connected(
        ratio_net,
        self._ratio_num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)
    
    net = slim.fully_connected(
        net, 512, weights_initializer=weights_initializer)
    net = slim.fully_connected(
        net,
        self.num_actions * self._num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)

    c_logits = tf.reshape(ratio_net, [-1, 1, self._ratio_num_atoms])
    c_probabilities = tf.contrib.layers.softmax(c_logits)
    c_values = tf.reduce_sum(self._ratio_support * c_probabilities, axis=2)

    logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(self._support * probabilities, axis=2)
    return self._get_network_type()(c_values, c_logits, c_probabilities, q_values, logits, probabilities)

  def _build_networks(self):
    """Extends the build networks computations with:

      self._replay_next_net_outputs: The replayed next states' values.
      self._replay_target_net_outputs: The replayed states' target
        values.
    """
    super(CovariateShiftAgent, self)._build_networks()

    self._replay_next_net_outputs = self.online_convnet(self._replay.u_next_states)
    self._replay_target_net_outputs = self.target_convnet(self._replay.u_states)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A `WrappedCSReplayBuffer` object.

    Raises:
      ValueError: if given an invalid replay scheme.
    """
    return cs_replay_buffer.WrappedCSReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma)

  def compute_policies_quotient(self):
    '''Computes the quotient of policies that appears at the DCOP update

    Returns:
      policies_quotient: tf.tensor, the quotient from the replay
    '''
    batch_size = self._replay.batch_size
    qt_argmax_actions = tf.argmax(self._replay_target_net_outputs.q_values, axis=1)[:, None]
    qt_argmax_actions = tf.reshape(qt_argmax_actions, [batch_size,1])
    replay_actions = tf.reshape(self._replay.actions, [batch_size,1])

    coincidences = tf.equal(qt_argmax_actions, replay_actions)

    coincidence_quotient = tf.fill([batch_size,1], self.num_actions * (1 - self.epsilon_eval))
    no_coincidence_quotient = tf.fill([batch_size,1], self.epsilon_eval)

    policies_quotient = tf.where(coincidences, coincidence_quotient, no_coincidence_quotient)
    return policies_quotient

  def _build_target_c_distribution(self):
    """Builds the c ratio target distribution.

    First, we compute the support of the target, gamma * (pi/mu) * x + (1-gamma), Where x
    is the support of the previous state distribution:

      * Evenly spaced in [vmin, vmax] if the current state is nonbegginer;
      * 1 otherwise (duplicated ratio_num_atoms times).

    Second, we compute the cs ratio probabilities of the current state.

    Finally we project the target (support + probabilities) onto the
    original support.

    Returns:
      target_distribution: tf.tensor, the target distribution from the replay.
    """
    batch_size = self._replay.batch_size

    # size of tiled_support: batch_size x ratio_num_atoms
    tiled_support = tf.tile(self._ratio_support, [batch_size])
    tiled_support = tf.reshape(tiled_support, [batch_size, self._ratio_num_atoms])
    # incorporate beginning states, whose tiled support is 1
    beginning_mask = tf.tile(tf.cast(self._replay.u_beginnings, tf.bool),[1, self._ratio_num_atoms])
    ones = tf.ones([batch_size, self._ratio_num_atoms])

    tiled_support = tf.where(beginning_mask, ones, tiled_support)
    # size of target_support: batch_size x ratio_num_atoms

    # Compute the quotient of policies
    policies_quotient = self.compute_policies_quotient()

    # Addition of the constant term, 1-discount factor
    constant_term = tf.tile(1. - self.ratio_discount_factor, [batch_size, self._ratio_num_atoms])

    target_support = self.ratio_discount_factor * policies_quotient * tiled_support + constant_term

    # size of next_probabilities: batch_size x ratio_num_atoms
    probabilities = self._replay_target_net_outputs.c_probabilities

    return rainbow_agent.project_distribution(target_support, probabilities,
                                              self._ratio_support)
  
  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    # Loss of the ratio model
    c_target_distribution = tf.stop_gradient(self._build_target_c_distribution())

    c_logits = self._replay_next_net_outputs.c_logits

    c_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=c_target_distribution,
        logits=c_logits)
    c_loss = tf.scalar_mul(self.ratio_loss_weight, c_loss)

    # Loss os the Q model
    target_distribution = tf.stop_gradient(self._build_target_distribution())

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_net_outputs.logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_net_outputs.logits,
                                        reshaped_actions)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_distribution,
        logits=chosen_action_logits)

    probs = self._replay.transition['sampling_probabilities']
    loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
    loss_weights /= tf.reduce_max(loss_weights)

    update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, self._replay_net_outputs.c_values)
    
    # Weight the loss by the inverse priorities.
    loss = loss_weights * loss

    # Generate the final loss
    final_loss = loss + c_loss

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('CrossEntropyLoss', tf.reduce_mean(final_loss))
      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      return self.optimizer.minimize(tf.reduce_mean(final_loss)), final_loss
