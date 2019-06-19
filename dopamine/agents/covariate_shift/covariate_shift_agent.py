# coding=utf-8
# gbg141

"""Agent performing a Distributional Discounted COP-TD

Details in "Off-Policy Deep Reinforcement Learning by Bootstrapping the Covariate Shift" by 
Carles Gelada & Marc G. Bellemare (2018)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import matplotlib
matplotlib.use('agg')
#from matplotlib import gridspec
import matplotlib.pyplot as plt

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.replay_memory import prioritized_replay_buffer, cs_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
#from dopamine.discrete_domains import atari_lib
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
               network=None,
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
               update_beginning_priorities=True,
               quotient_epsilon=0.1,
               quotient_epsilon_decay_period=1000000,
               use_loss_weights=False,
               ratio_num_atoms=101,
               ratio_cmin=0.05,
               ratio_cmax=5.,
               log_ratio_approach=False,
               use_ratio_exp_bins=False,
               plot_log_scale=False,
               define_base_and_exp=False,
               ratio_exp_base=2.,
               ratio_min_exp=None,
               ratio_max_exp=8,
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
      network: function expecting three parameters:
        (num_actions, network_type, state). This function will return the
        network_type object containing the tensors output by the network.
        See dopamine.discrete_domains.atari_lib.nature_dqn_network as
        an example. (UNUSED HERE)
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
      update_beginning_priorities: bool, whether to set beginning states to c=1
      quotient_epsilon: float, epsilon used when computing the quotient of policies
      quotient_epsilon_decay_period: int, length of the quotient epsilon decay schedule.
      use_loss_weights: bool, whether to use loss weights of the Q model
      ratio_num_atoms: int, the number of buckets of the ratio function distribution.
      ratio_cmin: float, the predefined minimum ratio value
      ratio_cmax: float, the predefined maximum ratio value
      log_ratio_approach: bool, whether to consider the logarithmic update approach
      use_ratio_exp_bins: bool, whether to use an exponential sequence of bins 
        instead of linear ones
      plot_log_scale: bool, whether to show x-axis distribution plot in log scale when
        using exponential bins
      define_base_and_exp: bool, whether the base and exponents are defined; otherwise,
        the default interval is divided in an exponential way
      ratio_exp_base: float, base of the exponential sequence
      ratio_min_exp: int, minimum exponent
      ratio_max_exp: int, maximum exponent
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
    self.update_beginning_priorities = update_beginning_priorities
    self.final_quotient_epsilon = quotient_epsilon
    self.quotient_epsilon_decay_period = quotient_epsilon_decay_period
    self.quotient_epsilon = tf.placeholder(tf.float32, name='quotient_epsilon_eff')
    self.use_loss_weights = use_loss_weights
    self.log_ratio_approach = log_ratio_approach
    self.use_ratio_exp_bins = use_ratio_exp_bins
    self.plot_log_scale = plot_log_scale
    self.define_base_and_exp = define_base_and_exp
    if self.use_ratio_exp_bins and not self.log_ratio_approach:
      if self.define_base_and_exp:
        self._ratio_exp_base = ratio_exp_base
        self._ratio_max_exp = ratio_max_exp
        self._ratio_min_exp = ratio_min_exp if ratio_min_exp is not None else -ratio_max_exp
        assert self._ratio_min_exp < 0
        self._ratio_support = tf.constant([float(ratio_exp_base**exp) for exp in 
                                          range(self._ratio_min_exp, self._ratio_max_exp+1)])
        self._log_ratio_support = tf.log(self._ratio_support) / tf.log(float(ratio_exp_base))
        self._ratio_num_atoms = self._ratio_max_exp - self._ratio_min_exp + 1
        self._ratio_cmin = float(self._ratio_exp_base**self._ratio_min_exp)
        self._ratio_cmax = float(self._ratio_exp_base**self._ratio_max_exp)
      else:
        self._ratio_num_atoms = ratio_num_atoms
        self._ratio_cmin = float(ratio_cmin)
        self._ratio_cmax = float(ratio_cmax)
        self._log_ratio_support = tf.linspace(float(np.log(self._ratio_cmin)), 
                                              float(np.log(self._ratio_cmax)), 
                                              ratio_num_atoms)
        self._ratio_support = tf.exp(self._log_ratio_support)
    else:
      self._ratio_num_atoms = ratio_num_atoms
      if self.log_ratio_approach:
        self._ratio_cmin = float(np.log(ratio_cmin))
        self._ratio_cmax = float(np.log(ratio_cmax))
        if self.use_ratio_exp_bins:
          self._log_ratio_support = tf.linspace(self._ratio_cmin, self._ratio_cmax, ratio_num_atoms)
          self._ratio_support = tf.exp(self._log_ratio_support)
        else:
          self._ratio_support = tf.linspace(float(ratio_cmin), float(ratio_cmax), ratio_num_atoms)
          self._log_ratio_support = tf.log(self._ratio_support)
      else:
        self._ratio_cmin = float(ratio_cmin)
        self._ratio_cmax = float(ratio_cmax)
        self._ratio_support = tf.linspace(self._ratio_cmin, self._ratio_cmax, ratio_num_atoms)
        self._log_ratio_support = tf.log(self._ratio_support)
    self.ratio_discount_factor = ratio_discount_factor
    self.ratio_loss_weight = ratio_loss_weight

    if self.use_ratio_model:
      tf.logging.info('Extra parameters of %s:', self.__class__.__name__)
      tf.logging.info('\t final_quotient_epsilon: %f', quotient_epsilon)
      tf.logging.info('\t quotient_epsilon_decay_period: %d', quotient_epsilon_decay_period)
      tf.logging.info('\t use_loss_weights: %s', use_loss_weights)
      tf.logging.info('\t log_ratio_approach: %s', log_ratio_approach)
      tf.logging.info('\t use_ratio_exp_bins: %s', self.use_ratio_exp_bins)
      if self.use_ratio_exp_bins and not log_ratio_approach:
        tf.logging.info('\t define_base_and_exp: %s', define_base_and_exp)
      tf.logging.info('\t ratio_num_atoms: %d', self._ratio_num_atoms)
      tf.logging.info('\t ratio_cmin: %f', self._ratio_cmin)
      tf.logging.info('\t ratio_cmax: %f', self._ratio_cmax)
      tf.logging.info('\t ratio_discount_factor: %f', ratio_discount_factor)
      tf.logging.info('\t ratio_loss_weight: %f', ratio_loss_weight)

    super(CovariateShiftAgent, self).__init__(
          sess=sess,
          num_actions=num_actions,
          observation_shape=observation_shape,
          observation_dtype=observation_dtype,
          stack_size=stack_size,
          network=network,
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

    TODO: Move this code to atari_lib and use 'network' parameter

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
      c_logits = tf.reshape(ratio_net, [-1, self._ratio_num_atoms], name='c_logits')
      c_probabilities = tf.contrib.layers.softmax(c_logits)
      c_values = tf.reduce_sum(self._ratio_support * c_probabilities, axis=1, name='c_values')

      logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms], name='q_logits')
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

    with tf.name_scope('u_replay'):
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

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    quotient_epsilon = self.epsilon_fn(
                        self.quotient_epsilon_decay_period,
                        self.training_steps,
                        self.min_replay_history,
                        self.final_quotient_epsilon)

    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sess.run(self._train_op, feed_dict={self.quotient_epsilon: quotient_epsilon})
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1

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
      
      coincidences = tf.equal(qt_argmax_actions, replay_actions, name='coincidences')

      #with tf.control_dependencies([initialize_epsilon]):
      coincidence_quotient = tf.fill([batch_size,1], self.num_actions * (1 - self.quotient_epsilon ) 
                                      + self.quotient_epsilon, 
                                      name='coincidence_value')
      no_coincidence_quotient = tf.fill([batch_size,1], self.quotient_epsilon,
                                          name='no_coincidence_value')

      policies_quotient = tf.where(coincidences, coincidence_quotient, no_coincidence_quotient, name='quotient')
    
    return policies_quotient

  def _build_target_c_distribution(self):
    """Builds the c ratio target distribution.

    First, we compute the support of the target, gamma * (pi/mu) * x + (1-gamma), Where x
    is the support of the previous state distribution:

      * Evenly spaced in [cmin, cmax] if the current state is nonbegginer;
      * 1 otherwise (replicated ratio_num_atoms times).

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
      tiled_support = tf.where(beginning_mask, tf.ones(tf.shape(tiled_support)), tiled_support, 
                               name='tiled_support')

      # Compute the quotient of policies
      policies_quotient = tf.tile(self.compute_policies_quotient(), [1, self._ratio_num_atoms],
                                  name='quotient')

      # Addition of the constant term, 1-discount factor
      constant_term = tf.fill([batch_size, self._ratio_num_atoms], 1. - self.ratio_discount_factor, 
                              name='constant_term')
      
      # target ratio
      target_ratio = tf.multiply(policies_quotient, tiled_support, name='target_ratio')

      # size of target_support: batch_size x ratio_num_atoms
      self._target_support  = tf.add(self.ratio_discount_factor * target_ratio, constant_term, 
                              name='target_support')

      # size of next_probabilities: batch_size x ratio_num_atoms
      probabilities = self._u_replay_target_net_outputs.c_probabilities

    return project_distribution(self._target_support, probabilities, self._ratio_support)

  def _build_target_c_distribution_with_log_approach(self):
    """Builds the c ratio target distribution using the logarithmic approach.

    First, we compute the support of the target, gamma * [log(pi/mu) + yi], Where yi,
    which represents the logarithm of the ratio values,
    is the support of the previous state distribution:

      * Evenly spaced in [cmin, cmax] if the current state is nonbegginer;
      * 0 otherwise (replicated ratio_num_atoms times).

    Second, we compute the cs ratio probabilities of the current state.

    Finally we project the target (support + probabilities) onto the
    original support.

    Returns:
      target_distribution: tf.tensor, the target distribution from the replay.
    """
    batch_size = self._replay.batch_size
    with tf.name_scope('c_target_distribution'):
      # size of tiled_support: batch_size x ratio_num_atoms
      log_tiled_support = tf.tile(self._log_ratio_support, [batch_size])
      log_tiled_support = tf.reshape(log_tiled_support, [batch_size, self._ratio_num_atoms])
      # incorporate beginning states, whose tiled support is 0
      beginning_mask = self._replay.uniform_transition['beginning']
      log_tiled_support = tf.where(beginning_mask, tf.zeros(tf.shape(log_tiled_support)), log_tiled_support, 
                               name='tiled_support')

      # Compute the log quotient of policies
      policies_quotient = tf.tile(self.compute_policies_quotient(), [1, self._ratio_num_atoms],
                                  name='quotient')
      log_policies_quotient = tf.log(policies_quotient, name='log_quotient')
      
      # target ratio
      log_target_ratio = tf.add(log_policies_quotient, log_tiled_support, name='log_target_ratio')

      # size of target_support: batch_size x ratio_num_atoms
      self._log_target_support  = tf.multiply(self.ratio_discount_factor, log_target_ratio,
                                              name='log_target_support')
      self._target_support = tf.exp(self._log_target_support, name='target_support')

      # size of next_probabilities: batch_size x ratio_num_atoms
      probabilities = self._u_replay_target_net_outputs.c_probabilities

    return project_distribution(self._log_target_support, probabilities, self._log_ratio_support)
  
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
        if self.log_ratio_approach:
          c_target_distribution = tf.stop_gradient(self._build_target_c_distribution_with_log_approach(), 
                                                   name='c_distribution')
        else:
          c_target_distribution = tf.stop_gradient(self._build_target_c_distribution(), 
                                                   name='c_distribution')
        self.c_target_distribution = c_target_distribution

        c_logits = self._u_replay_next_net_outputs.c_logits

        c_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=c_target_distribution,
            logits=c_logits,
            name='c_loss')

        # Avoid training with undefined next states (i.e. terminal states)
        terminal_mask = 1. - tf.cast(self._replay.uniform_transition['terminal'], tf.float32, name='terminal')
        c_loss = terminal_mask * c_loss

        # Generate the final loss
        final_loss = tf.add(loss, self.ratio_loss_weight * c_loss, name='final_loss')

        # Update priorities, being those of beginnings 1
        priorities = self._replay_net_outputs.c_values
        if self.update_beginning_priorities:
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
            tf.summary.scalar('MeanRatioLoss', tf.reduce_mean(c_loss))
            tf.summary.scalar('MeanQLoss', tf.reduce_mean(loss))
        if self.use_ratio_model:
          with tf.variable_scope('Priorities'):
            mean, var = tf.nn.moments(priorities, axes=[0])
            max_priority = tf.reduce_max(priorities)
            min_priority = tf.reduce_min(priorities)
            tf.summary.scalar('MeanPriorities', mean)
            tf.summary.scalar('VarPriorities', var)
            tf.summary.scalar('MaxPriorities', max_priority)
            tf.summary.scalar('MinPriorities', min_priority)
            tf.summary.text('Values', tf.as_string(priorities))
          with tf.variable_scope('Histograms'):
            tf.summary.histogram('c_dist', c_target_distribution[0,:])
            tf.summary.histogram('c_logits', c_logits[0,:])
            tf.summary.histogram('c_probs', self._u_replay_target_net_outputs.c_probabilities[0,:])
          with tf.variable_scope('Images_Qmodel'):
            argmax_c_value = tf.argmax(self._replay_net_outputs.c_values, axis=0)
            self.compute_c_distribution_summaries_qmodel(argmax_c_value, prefix_name='ARGMAX_')
            argmax_frame = self._replay.states[argmax_c_value:argmax_c_value+1,:,:,3:4]
            tf.summary.image('ARGMAX_Frame_Q', argmax_frame)

            argmin_c_value = tf.argmin(self._replay_net_outputs.c_values, axis=0)
            self.compute_c_distribution_summaries_qmodel(argmin_c_value, prefix_name='ARGMIN_')
            argmin_frame = self._replay.states[argmin_c_value:argmin_c_value+1,:,:,3:4]
            tf.summary.image('ARGMIN_Frame_Q', argmin_frame)
          with tf.variable_scope('Images_Cmodel'):
            argmax_c_value = tf.argmax(self._u_replay_next_net_outputs.c_values, axis=0)
            self.compute_c_distribution_summaries(argmax_c_value, prefix_name='ARGMAX_')
            argmax_frame = self._replay.u_next_states[argmax_c_value:argmax_c_value+1,:,:,3:4]
            tf.summary.image('ARGMAX_Frame', argmax_frame)

            argmin_c_value = tf.argmin(self._u_replay_next_net_outputs.c_values, axis=0)
            self.compute_c_distribution_summaries(argmin_c_value, prefix_name='ARGMIN_')
            argmin_frame = self._replay.u_next_states[argmin_c_value:argmin_c_value+1,:,:,3:4]
            tf.summary.image('ARGMIN_Frame', argmin_frame)
            
      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      return self.optimizer.minimize(tf.reduce_mean(final_loss)), final_loss

  def compute_c_distribution_summaries_qmodel(self, index, prefix_name=''):
    predicted_dist_support = self._ratio_support 
    self.c_distribution_summary(
      support=predicted_dist_support, 
      dist_values=self._replay_net_outputs.c_probabilities[index], 
      name=prefix_name+'Predicted_Dist_Q')
    
  def compute_c_distribution_summaries(self, index, prefix_name=''):
    predicted_dist_support = self._ratio_support 
    self.c_distribution_summary(
      support=predicted_dist_support, 
      dist_values=self._u_replay_next_net_outputs.c_probabilities[index], 
      name=prefix_name+'Predicted_Dist')

    target_dist_support = self._target_support[index]
    self.c_distribution_summary(
      support=target_dist_support, 
      dist_values=self._u_replay_target_net_outputs.c_probabilities[index], 
      name=prefix_name+'Target_Dist')

    projected_dist_support = self._ratio_support 
    self.c_distribution_summary(
      support=projected_dist_support, 
      dist_values=self.c_target_distribution[index], 
      name=prefix_name+'Projected_Dist') 
            
  def c_distribution_summary(self, support, dist_values, name=None):
    c_value = tf.reduce_sum(support*dist_values, axis=0)
    pred_dist = tf.py_func(
      self.plot_c_value_distribution, [c_value, support, dist_values, self.use_ratio_exp_bins, self.plot_log_scale],
      [tf.uint8],
      name=name
    )
    tf.summary.image(name, pred_dist)

  def plot_c_value_distribution(self, c_value, support, dist_values, use_exp_bins=False, log_scale=True): 
    num_atoms = support.shape[0]
    fig, ax = plt.subplots(figsize=(5, 3))
    if use_exp_bins:
      base = (support[-1]/support[0])**(1./num_atoms)
      extra_mark = support[0]*(base)**(num_atoms+1)
      if log_scale: ax.set_xscale('log')
    else:
      extra_mark = support[-1] + (support[-1]-support[0])/num_atoms
    width = np.diff(np.hstack((support, np.array((extra_mark)))))
    ax.set_title('c: ' + str(c_value))
    ax.set_xlabel('c value')
    ax.set_ylabel('probability')
    ax.bar(support, dist_values, width=width, alpha=0.5, align='edge',linewidth=1, edgecolor='black')
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    dist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    dist = dist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return dist

def project_distribution(supports, weights, target_support):
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  This code is not easy to digest, so we will use a running example to clarify
  what is going on, with the following sample inputs:

    * supports =       [[0, 2, 4, 6, 8],
                        [1, 3, 4, 5, 6]]
    * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.5, 0.1, 0.1]]
    * target_support = [4, 5, 6, 7, 8]

  In the code below, comments preceded with 'Ex:' will be referencing the above
  values.

  Args:
    supports: Tensor of shape (batch_size, num_dims) defining supports for the
      distribution.
    weights: Tensor of shape (batch_size, num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Tensor of shape (num_dims) defining support of the projected
      distribution. The values must be monotonically increasing. Vmin and Vmax
      will be inferred from the first and last elements of this tensor,
      respectively. The values in this tensor must be equally spaced.

  Returns:
    A Tensor of shape (batch_size, num_dims) with the projection of a batch of
    (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  target_support_deltas_forward = tf.pad(target_support[1:] - target_support[:-1], 
                                         tf.constant([[0,1]]), constant_values=1)
  target_support_deltas_backward = tf.pad(target_support[:-1] - target_support[1:],
                                          tf.constant([[1,0]]), constant_values=-1)
  
  validate_deps = []
  supports.shape.assert_is_compatible_with(weights.shape)
  supports[0].shape.assert_is_compatible_with(target_support.shape)
  target_support.shape.assert_has_rank(1)

  with tf.control_dependencies(validate_deps):
    v_min, v_max = target_support[0], target_support[-1]
    batch_size = tf.shape(supports)[0]
    num_dims = tf.shape(target_support)[0]

    clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
    tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])

    reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
    reshaped_target_support = tf.reshape(reshaped_target_support,
                                        [batch_size, num_dims, 1])

    reshaped_target_support_deltas_forward = tf.tile(target_support_deltas_forward[:, None], 
                                                     [batch_size, num_dims])
    reshaped_target_support_deltas_forward = tf.reshape(reshaped_target_support_deltas_forward,
                                                        [batch_size, num_dims, num_dims])
    
    reshaped_target_support_deltas_backward = tf.tile(target_support_deltas_backward[:, None], 
                                                      [batch_size, num_dims])
    reshaped_target_support_deltas_backward = tf.reshape(reshaped_target_support_deltas_backward,
                                                         [batch_size, num_dims, num_dims])
    
    numerator = tiled_support - reshaped_target_support
    numerator_sign_mask = numerator[0] <= 0

    reshaped_target_support_deltas = tf.where(numerator_sign_mask, 
                                              reshaped_target_support_deltas_backward,
                                              reshaped_target_support_deltas_forward)

    quotient = 1 - (numerator / reshaped_target_support_deltas)
    clipped_quotient = tf.clip_by_value(quotient, 0, 1)

    weights = weights[:, None, :]
    inner_prod = clipped_quotient * weights

    projection = tf.reduce_sum(inner_prod, 3)
    projection = tf.reshape(projection, [batch_size, num_dims])
    return projection
