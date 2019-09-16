<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cs_replay_buffer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cs_replay_buffer

An extension of Prioritized Experience Replay (PER) to work with Covariate Shift Ratio estimates.

## Classes

[`class WrappedCSReplayBuffer`](./cs_replay_buffer/WrappedCSReplayBuffer.md):
Wrapper of [`OutOfGraphPrioritizedReplayBuffer`](./prioritized_replay_buffer/OutOfGraphPrioritizedReplayBuffer.md), with both uniform and
  in-graph sampling, that also manages the storage and sampling of Covariate Shift Ratios.