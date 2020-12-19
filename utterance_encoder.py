#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : utterance_encoder.py
# Author            : Yan <yanwong@126.com>
# Date              : 01.12.2020
# Last Modified Date: 19.12.2020
# Last Modified By  : Yan <yanwong@126.com>

import tensorflow as tf

class PretrainedEmbedding(tf.keras.layers.Layer):
  def __init__(self, embedding):
    super(PretrainedEmbedding, self).__init__()

    self.embedding = tf.constant(embedding)

  def call(self, x):
    x = tf.nn.embedding_lookup(self.embedding, x)
    return x

class CnnUtteranceEncoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, filters, kernel_sizes,
               d_word, d_sent, rate=0.1, embedding=None):
    super(CnnUtteranceEncoder, self).__init__()

    if embedding is None:
      self.embedding = tf.keras.layers.Embedding(vocab_size, d_word)
    else:
      self.embedding = embedding

    self.convs = [tf.keras.layers.Conv1D(filters, h, activation='relu')
                  for i, h in enumerate(kernel_sizes)]

    self.dense = tf.keras.layers.Dense(d_sent, activation='relu')
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask=None):
    # x.shape == (batch_size, seq_len)

    x = self.embedding(x)
    # each has shape of (batch_size, seq_len - kernel_size + 1, filters)
    conved = [conv(x) for conv in self.convs]
    # each has shape of (batch_size, 1, filters)
    pooled = [tf.nn.max_pool1d(c, c.shape[1], c.shape[1], 'VALID') for c in conved]
    # each has shape of (batch_size, filters)
    pooled = [tf.squeeze(p) for p in pooled]
    pooled = tf.concat(pooled, axis=1)  # (batch_size, len(pooled) * filters)
    
    feature = self.dense(self.dropout(pooled))  # (batch_size, d_sent)

    return feature

