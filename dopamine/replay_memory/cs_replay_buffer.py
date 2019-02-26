# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of Prioritized Experience Replay (PER).

This implementation is based on the paper "Prioritized Experience Replay"
by Tom Schaul et al. (2015). Many thanks to Tom Schaul, John Quan, and Matteo
Hessel for providing useful pointers on the algorithm and its implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip

from dopamine.replay_memory import circular_replay_buffer, prioritized_replay_buffer
from dopamine.replay_memory import sum_tree
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
import numpy as np
import tensorflow as tf

import gin.tf


@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedCSReplayBuffer(
      prioritized_replay_buffer.WrappedPrioritizedReplayBuffer):
  """Wrapper of OutOfGraphPrioritizedReplayBuffer with both uniform and
  in-graph sampling.

  Usage:

    * To add a transition:  Call the add function.

    * To sample a batch:  Query any of the tensors in the transition dictionary.
                          Every sess.run that requires any of these tensors will
                          sample a new transition.
  """

  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=circular_replay_buffer.MAX_SAMPLE_ATTEMPTS,
               extra_storage_types=None,
               observation_dtype=np.uint8):
    """Initializes WrappedPrioritizedReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    super(WrappedCSReplayBuffer, self).__init__(
        observation_shape,
        stack_size,
        use_staging,
        replay_capacity,
        batch_size,
        update_horizon,
        gamma,
        extra_storage_types=extra_storage_types)

  def create_sampling_ops(self, use_staging):
    """Creates the ops necessary to sample from the replay buffer.

    Creates the transition dictionary containing the sampling tensors.

    Args:
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
    """
    with tf.name_scope('sample_replay'):
      with tf.device('/cpu:*'):
        transition_type = self.memory.get_transition_elements()
        transition_tensors = tf.py_func(
            self.memory.sample_transition_batch, [],
            [return_entry.type for return_entry in transition_type],
            name='replay_sample_py_func')
        uniform_sample_indices = tf.py_func(
            super(prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer, self.memory).
            sample_index_batch, [self.batch_size], [np.int64 for _ in range(self.batch_size)],
            name='uniform_index_sampling') 
        uniform_transition_tensors = tf.py_func(
            self.memory.sample_transition_batch, [self.batch_size, uniform_sample_indices],
            [return_entry.type for return_entry in transition_type],
            name='uniform_replay_sample_py_func')
        self._set_transition_shape(transition_tensors, transition_type)
        self._set_transition_shape(uniform_transition_tensors, transition_type)
        if use_staging:
          transition_tensors, uniform_transition_tensors = self._set_up_staging(transition_tensors, uniform_transition_tensors)
          self._set_transition_shape(transition_tensors, transition_type)
          self._set_transition_shape(uniform_transition_tensors, transition_type)

        # Unpack sample transition into member variables.
        self.unpack_transition(transition_tensors, uniform_transition_tensors, transition_type)

  def _set_up_staging(self, transition, u_transition):
    """Sets up staging ops for prefetching the next transition.

    This allows us to hide the py_func latency. To do so we use a staging area
    to pre-fetch the next batch of transitions.

    Args:
      transition: tuple of tf.Tensors with shape
        memory.get_transition_elements().
      u_transition: tuple of tf.Tensors with shape
        memory.get_transition_elements().

    Returns:
      prefetched_transition: tuple of tf.Tensors with shape
        memory.get_transition_elements() that have been previously prefetched.
    """
    transition_type = self.memory.get_transition_elements()

    # Create the staging area in CPU.
    prefetch_area = tf.contrib.staging.StagingArea(
        [shape_with_type.type for shape_with_type in transition_type])
    u_prefetch_area = tf.contrib.staging.StagingArea(
        [shape_with_type.type for shape_with_type in transition_type])

    # Store prefetch op for tests, but keep it private -- users should not be
    # calling _prefetch_batch.
    self._prefetch_batch = prefetch_area.put(transition)
    initial_prefetch = tf.cond(
        tf.equal(prefetch_area.size(), 0),
        lambda: prefetch_area.put(transition), tf.no_op)
    
    self._u_prefetch_batch = u_prefetch_area.put(transition)
    u_initial_prefetch = tf.cond(
        tf.equal(u_prefetch_area.size(), 0),
        lambda: u_prefetch_area.put(u_transition), tf.no_op)

    # Every time a transition is sampled self.prefetch_batch will be
    # called. If the staging area is empty, two put ops will be called.
    with tf.control_dependencies([self._prefetch_batch, initial_prefetch, 
                                  self._u_prefetch_batch, u_initial_prefetch]):
      prefetched_transition = prefetch_area.get()
      u_prefetched_transition = u_prefetch_area.get()

    return prefetched_transition, u_prefetched_transition

  def unpack_transition(self, transition_tensors, uniform_transition_tensors, transition_type):
    """Unpacks the given transition into member variables.

    Args:
      transition_tensors: tuple of tf.Tensors.
      uniform_transition_tensors: tuple of tf.Tensors.
      transition_type: tuple of ReplayElements matching transition_tensors.
    """
    self.transition = collections.OrderedDict()
    for element, element_type in zip(transition_tensors, transition_type):
      self.transition[element_type.name] = element

    self.states = self.transition['state']
    self.actions = self.transition['action']
    self.rewards = self.transition['reward']
    self.next_states = self.transition['next_state']
    self.terminals = self.transition['terminal']
    self.indices = self.transition['indices']

    self.uniform_transition = collections.OrderedDict()
    for element, element_type in zip(uniform_transition_tensors, transition_type):
      self.uniform_transition[element_type.name] = element

    self.u_states = self.uniform_transition['state']
    self.u_actions = self.uniform_transition['action']
    self.u_rewards = self.uniform_transition['reward']
    self.u_next_states = self.uniform_transition['next_state']
    self.u_terminals = self.uniform_transition['terminal']
    self.u_indices = self.uniform_transition['indices']
