#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : configuration.py
# Author            : Yan <yanwong@126.com>
# Date              : 17.12.2020
# Last Modified Date: 08.01.2021
# Last Modified By  : Yan <yanwong@126.com>

class ModelConfig:
  def __init__(self):
    self.d_word = 300
    self.d_sent = 100
    self.filters = 50
    self.kernel_sizes = [3, 4, 5]
    self.n_classes = 6
    self.recur_units = 100
    self.d_context = 100
    self.recur_dropout = 0.2

class TrainingConfig:
  def __init__(self):
    self.learning_rate = 0.0001
    self.batch_size = 64
    # self.clip_gradients = 5.
    self.n_epochs = 100
    # self.freq_val = 100
    self.loss_weights = [
        1/0.086747,
        1/0.144406,
        1/0.227883,
        1/0.160585,
        1/0.127711,
        1/0.252668]

