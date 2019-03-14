# coding=utf-8
# gbg141

"""Agent permorming a Distributional Discounted COP-TD

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
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
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
               use_ratio_model=True,
               use_priorities=True,
               quotient_epsilon=0.1,
               use_loss_weights=False,
               ratio_num_atoms=51,
               ratio_cmin=None,
               ratio_cmax=None,
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
      use_ratio_model: bool, whether to train the ratio model or not,
      use_priorities: bool, whether to use priorities from the ratio model
      quotient_epsilon: float, epsilon used when computing the quotient of policies
      use_loss_weights: bool, whether to use loss weights of the Q model
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
    self.use_ratio_model = use_ratio_model
    self.use_priorities = use_priorities
    self.quotient_epsilon = quotient_epsilon
    self.use_loss_weights = use_loss_weights
    ratio_cmin = float(ratio_cmin) if ratio_cmin is not None else self.quotient_epsilon
    ratio_cmax = float(ratio_cmax) if ratio_cmax is not None else float(num_actions)
    self._ratio_num_atoms = ratio_num_atoms
    self._ratio_support = tf.linspace(ratio_cmin, ratio_cmax, ratio_num_atoms)
    self.ratio_discount_factor = ratio_discount_factor
    self.ratio_loss_weight = ratio_loss_weight

    if self.use_ratio_model:
      tf.logging.info('Extra parameters of %s:', self.__class__.__name__)
      tf.logging.info('\t quotient_epsilon: %f', quotient_epsilon)
      tf.logging.info('\t use_loss_weights: %s', use_loss_weights)
      tf.logging.info('\t ratio_num_atoms: %d', ratio_num_atoms)
      tf.logging.info('\t ratio_cmin: %f', ratio_cmin)
      tf.logging.info('\t ratio_cmax: %f', ratio_cmax)
      tf.logging.info('\t ratio_discount_factor: %f', ratio_discount_factor)
      tf.logging.info('\t ratio_loss_weight: %f', ratio_loss_weight)

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
    
    Now the internal state is_beginning is added.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    action = super(CovariateShiftAgent, self).begin_episode(observation)
    self.is_beginning = True
    return action

  def _store_transition(self, last_observation, action, reward, is_terminal, 
                        is_beginning=None, priority=None):
    """Stores an experienced transition, extended by the internal state is_beginning.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    is_beginning = is_beginning if is_beginning else self.is_beginning
    if priority is None:
      priority = self._replay.memory.sum_tree.max_recorded_priority
    
    self._replay.add(last_observation, action, reward, is_terminal, is_beginning, priority)
    self.is_beginning = False

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

    with tf.name_scope('network_outputs'):
      c_logits = tf.reshape(ratio_net, [-1, self._ratio_num_atoms])
      c_probabilities = tf.contrib.layers.softmax(c_logits)
      c_values = tf.reduce_sum(self._ratio_support * c_probabilities, axis=1, name='c_values')

      logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
      probabilities = tf.contrib.layers.softmax(logits)
      q_values = tf.reduce_sum(self._support * probabilities, axis=2, name='q_values')
    return self._get_network_type()(c_values, c_logits, c_probabilities, q_values, logits, probabilities)

  def _build_networks(self):
    """Extends the build networks computations with:

      self._replay_next_net_outputs: The replayed next states' values.
      self._replay_target_net_outputs: The replayed states' target
        values.
    """
    super(CovariateShiftAgent, self)._build_networks()

    self._u_replay_next_net_outputs = self.online_convnet(self._replay.u_next_states)
    self._u_replay_target_net_outputs = self.target_convnet(self._replay.u_states)

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
        gamma=self.gamma,
        extra_storage_types=[ReplayElement('beginning', (), np.bool)])

  def compute_policies_quotient(self):
    '''Computes the quotient of policies that appears at the DCOP update

    Returns:
      policies_quotient: tf.tensor, the quotient from the replay
    '''
    batch_size = self._replay.batch_size

    with tf.name_scope('policies_quotient'):
      qt_argmax_actions = tf.argmax(self._u_replay_target_net_outputs.q_values, 
                                    axis=1, output_type=tf.int32, name='qt_argmax_actions')
      replay_actions = self._replay.actions
      coincidences = tf.equal(qt_argmax_actions, replay_actions)

      coincidence_quotient = tf.fill([batch_size,1], self.num_actions * (1 - self.quotient_epsilon))
      no_coincidence_quotient = tf.fill([batch_size,1], self.quotient_epsilon)

      policies_quotient = tf.where(coincidences, coincidence_quotient, no_coincidence_quotient, name='quotient')
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
    with tf.name_scope('c_target_distribution'):
      # size of tiled_support: batch_size x ratio_num_atoms
      tiled_support = tf.tile(self._ratio_support, [batch_size])
      tiled_support = tf.reshape(tiled_support, [batch_size, self._ratio_num_atoms])
      # incorporate beginning states, whose tiled support is 1
      beginning_mask = self._replay.uniform_transition['beginning']
      tiled_support = tf.where(beginning_mask, tf.ones(tf.shape(tiled_support)), tiled_support, name='tiled_support')

      # Compute the quotient of policies
      policies_quotient = self.compute_policies_quotient()

      # Addition of the constant term, 1-discount factor
      constant_term = tf.fill([batch_size, self._ratio_num_atoms], 1. - self.ratio_discount_factor)

      # size of target_support: batch_size x ratio_num_atoms
      target_support = tf.identity(self.ratio_discount_factor * policies_quotient * tiled_support + constant_term, 
                                   name='target_support')

      # size of next_probabilities: batch_size x ratio_num_atoms
      probabilities = tf.identity(self._u_replay_target_net_outputs.c_probabilities,
                                  name='probabilities')

    return rainbow_agent.project_distribution(target_support, probabilities, self._ratio_support)
  
  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """

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

    if self.use_loss_weights:
      # Weight the loss by the inverse priorities.
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)
      loss = loss_weights * loss

    if self.use_ratio_model:
      # Loss of the ratio model
      with tf.name_scope('c_train_op'):
        c_target_distribution = tf.stop_gradient(self._build_target_c_distribution(), name='c_distribution')

        c_logits = tf.identity(self._u_replay_next_net_outputs.c_logits, name='c_logits')

        c_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=c_target_distribution,
            logits=c_logits,
            name='c_loss')

        # Avoid training with undefined next states (i.e. terminal states)
        terminal_mask = 1. - tf.cast(self._replay.uniform_transition['terminal'], tf.float32, name='terminal')
        c_loss = terminal_mask * c_loss

        # Generate the final loss
        final_loss = loss + self.ratio_loss_weight * c_loss

        # Update priorities, being those of beginnings 1
        priorities = self._replay_net_outputs.c_values
        beginning_mask = self._replay.transition['beginning']
        priorities = tf.where(beginning_mask, tf.ones(tf.shape(priorities)), priorities, name='priorities')
        if self.use_priorities:
          update_priorities_op = self._replay.tf_set_priority(
                self._replay.indices, priorities)
        else:
          update_priorities_op = tf.no_op()  
    else:
      final_loss = loss
      update_priorities_op = tf.no_op()
    
    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('CrossEntropyLoss', tf.reduce_mean(final_loss))
          if self.use_ratio_model:
            tf.summary.scalar('RatioLoss', tf.reduce_mean(c_loss))
            tf.summary.scalar('QLoss', tf.reduce_mean(loss))
        if self.use_ratio_model:
          with tf.variable_scope('CSratio'):
            tf.summary.scalar('MeanRatioValues', tf.reduce_mean(priorities))
          with tf.variable_scope('Masks'):
            tf.summary.scalar('BeginningMask', tf.reduce_sum(tf.cast(beginning_mask, tf.int8)))
            tf.summary.scalar('TerminalMask', tf.reduce_sum(terminal_mask))
      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      return self.optimizer.minimize(tf.reduce_mean(final_loss)), final_loss
