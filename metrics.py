#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : metrics.py
# Author            : Yan <yanwong@126.com>
# Date              : 26.12.2020
# Last Modified Date: 29.12.2020
# Last Modified By  : Yan <yanwong@126.com>

import tensorflow as tf

def binarize_classification(classification, class_id):
  binary = tf.math.equal(classification, class_id)
  return tf.cast(binary, dtype=tf.float32)

class ConfusionMatrix(tf.keras.metrics.Metric):
  def __init__(self, n_classes, name='confusion_matrix', **kwargs):
    super(ConfusionMatrix, self).__init__(name=name, **kwargs)

    self.n_classes = n_classes
    self.confusion_matrix = self.add_weight(name='cmat',
                                            shape=(n_classes, 4),
                                            initializer="zeros")

  def update_state(self, y_true, y_pred, sample_weight=None):
    cmat = []
    for i in range(self.n_classes):
      binary_y_true = binarize_classification(y_true, i)
      binary_y_pred = binarize_classification(y_pred, i)

      tp_vals = binary_y_pred * binary_y_true
      fp_vals = binary_y_pred * (1 - binary_y_true)
      tn_vals = (1 - binary_y_pred) * (1 - binary_y_true)
      fn_vals = (1 - binary_y_pred) * binary_y_true

      if sample_weight is not None:
        tp_vals = tf.math.multiply(tp_vals, sample_weight)
        fp_vals = tf.math.multiply(fp_vals, sample_weight)
        tn_vals = tf.math.multiply(tn_vals, sample_weight)
        fn_vals = tf.math.multiply(fn_vals, sample_weight)

      tp = tf.math.reduce_sum(tp_vals)
      fp = tf.math.reduce_sum(fp_vals)
      tn = tf.math.reduce_sum(tn_vals)
      fn = tf.math.reduce_sum(fn_vals)

      cmat.append([tp, fp, tn, fn])

    cmat = tf.stack(cmat)
    self.confusion_matrix.assign_add(cmat)

  def result(self):
    return self.confusion_matrix

  def reset_states(self):
    self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

def classification_report(confusion_matrix):
  cmat = confusion_matrix.result()  # (n_classes, 4)

  precision = tf.math.divide_no_nan(cmat[:, 0],
                                    cmat[:, 0] + cmat[:, 1])
  support = cmat[:, 0] + cmat[:, 3]
  recall = tf.math.divide_no_nan(cmat[:, 0], support)
  f1_score = tf.math.divide_no_nan(2 * precision * recall,
                                   precision + recall)
  accuracy = tf.math.divide_no_nan(cmat[:, 0] + cmat[:, 2],
                                   tf.reduce_sum(cmat, axis=1))

  class_report = tf.concat([
    tf.expand_dims(precision, -1),
    tf.expand_dims(recall, -1),
    tf.expand_dims(f1_score, -1),
    tf.expand_dims(accuracy, -1),
    tf.expand_dims(support, -1)], axis=1)  # (n_classes, 5)

  macro_f1_score = tf.math.reduce_mean(f1_score)

  total = tf.math.reduce_sum(cmat, axis=-1)
  weight = support / total

  weighted_f1_score = tf.math.reduce_sum(weight * f1_score)

  total_tp = tf.math.reduce_sum(cmat[:, 0])
  total_fp = tf.math.reduce_sum(cmat[:, 1])
  total_tn = tf.math.reduce_sum(cmat[:, 2])
  total_fn = tf.math.reduce_sum(cmat[:, 3])

  total_accuracy = tf.math.divide(total_tp,
                                  tf.math.reduce_sum(support))
  total_precision = tf.math.divide_no_nan(total_tp, total_tp + total_fp)
  total_recall = tf.math.divide_no_nan(total_tp, total_tp + total_fn)

  micro_f1_score = tf.math.divide_no_nan(2 * total_precision * total_recall,
                                         total_precision + total_recall)

  return class_report, micro_f1_score, macro_f1_score, weighted_f1_score, total_accuracy

# metric = ConfusionMatrix(3)
# metric.update_state([0, 1, 2, 2, 2], [0, 0, 2, 2, 1])
# print(metric.result().numpy())
# 
# class_report, accuracy, micro_f1, macro_f1, weighted_f1 = classification_report(metric)
# print(class_report, accuracy, micro_f1, macro_f1, weighted_f1)

# metric.update_state([1, 0, 0, 1, 2], [0, 0, 2, 1, 2])
# print(metric.result().numpy())
# 
# metric.reset_states()
# metric.update_state([0, 1, 2, 2, 2], [0, 0, 2, 2, 1], sample_weight=[1, 1, 1, 1, 0])
# print(metric.result().numpy())

