#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : emotion_model.py
# Author            : Yan <yanwong@126.com>
# Date              : 03.12.2020
# Last Modified Date: 08.01.2021
# Last Modified By  : Yan <yanwong@126.com>

import tensorflow as tf

def extract_utterance_features(utters, training, mask, utter_encoder):
  # utters.shape == (batch_size, dial_len, sent_len)

  batch_size, dial_len, sent_len = utters.shape
  utters = tf.reshape(utters, [-1, sent_len])  # (batch_size, dial_len * sent_len)
  feat = utter_encoder(utters, training, mask)  # (batch_size * dial_len, d_sent)
  feat = tf.reshape(feat, [batch_size, dial_len, -1])  # (batch_size, dial_len, d_sent)
  return feat

class ContextFreeModel(tf.keras.Model):
  def __init__(self, utter_encoder, n_classes):
    super(ContextFreeModel, self).__init__()

    self.utterance_encoder = utter_encoder
    self.dense = tf.keras.layers.Dense(
        n_classes,
        activation='relu',
        kernel_initializer=tf.keras.initializers.he_uniform())

  def call(self, x, training, mask):
    # mask.shape == (batch_size, dial_len, sent_len)

    if self.utterance_encoder is not None:
      # x.shape == (batch_size, dial_len, sent_len)

      batch_size, dial_len, sent_len = x.shape

      x = tf.reshape(x, [-1, sent_len])  # (batch_size * dial_len, sent_len)
      x = self.utterance_encoder(x, training, mask)  # (batch_size * dial_len, d_sent)
      x = tf.reshape(x, [batch_size, dial_len, -1])  # (batch_size, dial_len, d_sent)
    
    # x.shape == (batch_size, dial_len, d_sent)
    logits = self.dense(x)

    return logits

class CnnModel(tf.keras.Model):
  def __init__(self, vocab_size, filters, kernel_sizes,
               d_word, n_classes, rate=0.1, embedding=None):
    super(CnnModel, self).__init__()

    if embedding is not None:
      self.embedding = tf.keras.layers.Embedding(
          vocab_size,
          d_word,
          embeddings_initializer=tf.keras.initializers.Constant(embedding))
    else:
      self.embedding = tf.keras.layers.Embedding(vocab_size, d_word)

    self.convs = [tf.keras.layers.Conv1D(filters, h, activation='relu')
                  for i, h in enumerate(kernel_sizes)]

    self.dense = tf.keras.layers.Dense(
        n_classes,
        activation='relu',
        kernel_initializer=tf.keras.initializers.he_uniform())
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask=None):
    # x.shape == (batch_size, dial_len, sent_len)
    # mask.shape == (batch_szie, dial_len, sent_len)

    batch_size, dial_len, sent_len = x.shape

    x = tf.reshape(x, [-1, sent_len])  # (batch_size * dial_len, sent_len)

    x = self.embedding(x)
    # each has shape of (batch_size * dial_len, seq_len - kernel_size + 1, filters)
    conved = [conv(x) for conv in self.convs]
    # each has shape of (batch_size * dial_len, 1, filters)
    pooled = [tf.nn.max_pool1d(c, c.shape[1], c.shape[1], 'VALID') for c in conved]
    # each has shape of (batch_size * dial_len, filters)
    pooled = [tf.squeeze(p) for p in pooled]
    pooled = tf.concat(pooled, axis=1)  # (batch_size * dial_len, len(pooled) * filters)
    
    # logits.shape == (batch_size * dial_len, n_classes)
    logits = self.dense(self.dropout(pooled, training=training))

    return tf.reshape(logits, [-1, dial_len, n_classes])

class BiLstmModel(tf.keras.Model):
  def __init__(self, utter_encoder, recur_units, dff, dropout, n_classes):
    super(BiLstmModel, self).__init__()

    self.utterance_encoder = utter_encoder
    self.lstm1 = tf.keras.layers.LSTM(recur_units,
                                      return_sequences=True,
                                      recurrent_dropout=dropout)
    self.lstm2 = tf.keras.layers.LSTM(recur_units,
                                      return_sequences=True,
                                      recurrent_dropout=dropout)
    self.bilstm1 = tf.keras.layers.Bidirectional(self.lstm1)
    self.bilstm2 = tf.keras.layers.Bidirectional(self.lstm2)
    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(n_classes, activation='relu')
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, x, training, mask):
    # mask.shape == (batch_size, dial_len, sent_len)

    if self.utterance_encoder is not None:
      # x.shape == (batch_size, dial_len, sent_len)

      batch_size, dial_len, sent_len = x.shape

      x = tf.reshape(x, [-1, sent_len])  # (batch_size * dial_len, sent_len)
      x = self.utterance_encoder(x, training, mask)  # (batch_size * dial_len, d_sent)
      x = tf.reshape(x, [batch_size, dial_len, -1])  # (batch_size, dial_len, d_sent)

    mask = tf.math.reduce_sum(mask, axis=2)  # (batch_size, dial_len)
    mask = tf.cast(tf.math.not_equal(mask, 0), dtype=tf.float32)

    x = self.bilstm1(x, training=training, mask=mask)  # (batch_size, dial_len, 2*units)
    x = self.bilstm2(x, training=training, mask=mask)  # (batch_size, dial_len, 2*units)
    x = self.dense1(self.dropout(x, training=training))  # (batch_size, dial_len, dff)
    logits = self.dense2(x)

    return logits

