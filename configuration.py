#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : configuration.py
# Author            : Yan <yanwong@126.com>
# Date              : 17.12.2020
# Last Modified Date: 21.12.2020
# Last Modified By  : Yan <yanwong@126.com>

class ModelConfig:
  def __init__(self):
    self.d_word = 300
    self.d_sent = 100
    self.filters = 50
    self.kernel_sizes = [3, 4, 5]
    self.n_classes = 6

class TrainingConfig:
  def __init__(self):
    self.learning_rate = 0.001
    self.batch_size = 64
    # self.clip_gradients = 5.
    self.n_epochs = 100
    self.freq_val = 100

