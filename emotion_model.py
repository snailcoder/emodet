#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : emotion_model.py
# Author            : Yan <yanwong@126.com>
# Date              : 03.12.2020
# Last Modified Date: 21.12.2020
# Last Modified By  : Yan <yanwong@126.com>

import tensorflow as tf

def extract_utterance_features(utters, training, mask, utter_encoder):
  # utters.shape == (batch_size, dial_len, sent_len)

  batch_size, dial_len, sent_len = utters.shape
  utters = tf.reshape(utters, [-1, sent_len])  # (batch_size, dial_len * sent_len)
  feat = utter_encoder(utters, training, mask)  # (batch_size * dial_len, d_sent)
  feat = tf.reshape(feat, [batch_size, dial_len, -1])  # (batch_size, dial_len, d_sent)
  return feat

class ContextFreeModel(tf.keras.layers.Layer):
  def __init__(self, utter_encoder, n_classes):
    super(ContextFreeModel, self).__init__()

    self.utterance_encoder = utter_encoder
    self.dense = tf.keras.layers.Dense(n_classes, activation='relu')

  def call(self, x, training, mask):
    # x.shape == (batch_size, dial_len, sent_len)
    # mask.shape == (batch_szie, dial_len, sent_len)

    batch_size, dial_len, sent_len = x.shape

    x = tf.reshape(x, [-1, sent_len])  # (batch_size, dial_len * sent_len)
    x = self.utterance_encoder(x, training, mask)  # (batch_size * dial_len, d_sent)
    x = tf.reshape(x, [batch_size, dial_len, -1])  # (batch_size, dial_len, d_sent)
    
    logits = self.dense(x)
    pred = tf.nn.softmax(logits)

    return pred

class BiLstmModel(tf.keras.layers.Layer):
  def __init__(self, utter_encoder, units, rate, n_classes):
    super(BiLstmModel, self).__init__()

    self.utterance_encoder = utter_encoder
    self.lstm = tf.keras.layers.LSTM(units,
                                     return_sequences=True,
                                     recurrent_dropout=rate)
    self.bilstm = tf.keras.layers.Bidirectional(self.lstm)
    self.dense = tf.keras.layers.Dense(n_classes, activation='relu')

  def call(self, x, training, mask):
    # x.shape == (batch_size, dial_len, sent_len)
    # mask.shape == (batch_size, dial_len, sent_len)

    x = extract_utterance_features(x, training, mask, self.utterance_encoder)

    mask = tf.math.reduce_sum(mask, axis=2)  # (batch_size, dial_len)
    mask = tf.cast(tf.math.not_equal(mask, 0), dtype=tf.float32)

    x = self.bilstm(x, training=training, mask=mask)  # (batch_size, dial_len, 2*units)
    logits = self.dense(x)
    pred = tf.nn.softmax(logits)

    return pred

