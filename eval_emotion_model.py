#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : eval_emotion_model.py
# Author            : Yan <yanwong@126.com>
# Date              : 30.12.2020
# Last Modified Date: 08.01.2021
# Last Modified By  : Yan <yanwong@126.com>

import argparse

import numpy as np
import tensorflow as tf

import data_utils
import utterance_encoder
import emotion_model
import configuration
import metrics

parser = argparse.ArgumentParser(description='Evaluate emotion detection model.')

parser.add_argument('test_dataset', help='The path of test dataset.')
parser.add_argument('vocab', help='The vocab file.')
parser.add_argument('utterance_encoder',
                    help='The directory containing the checkpoint of utterance encoder.')
parser.add_argument('emotion_model',
                    help='The directory saving checkpoint of emotion model.')

args = parser.parse_args()

vocab = data_utils.load_vocab(args.vocab)

model_config = configuration.ModelConfig()
utter_encoder = utterance_encoder.CnnUtteranceEncoder(
    len(vocab),
    model_config.filters,
    model_config.kernel_sizes,
    model_config.d_word,
    model_config.d_sent)
utter_model = emotion_model.ContextFreeModel(utter_encoder, model_config.n_classes)
utter_model.load_weights(args.utterance_encoder)

model = emotion_model.BiLstmModel(None,
                                  model_config.recur_units,
                                  model_config.d_context,
                                  model_config.recur_dropout,
                                  model_config.n_classes)
model.load_weights(args.emotion_model)

def encode_utterance(utterance):
  batch_size, dial_len, sent_len = utterance.shape

  utterance = tf.reshape(utterance, [-1, sent_len])
  utterance = utter_model.utterance_encoder(utterance, False, None)

  return tf.reshape(utterance, [batch_size, dial_len, -1])

def evaluate(test_dataset):
  confusion_matrix = metrics.ConfusionMatrix(model_config.n_classes)

  for (batch, (speaker, utterance, emotion)) in enumerate(test_dataset):
    speaker = tf.squeeze(speaker)  # (batch_size, dial_len)
    emotion = tf.squeeze(emotion)  # (batch_size, dial_len)

    mask = tf.cast(tf.math.not_equal(utterance, 0), dtype=tf.float32)

    utterance = encode_utterance(utterance)

    predictions = model(utterance, False, mask)  # (batch_size, dial_len, n_classes)

    sample_weight = tf.math.not_equal(tf.math.reduce_sum(mask, axis=2), 0)
    sample_weight = tf.cast(sample_weight, dtype=tf.float32)
    pred_emotion = tf.math.argmax(predictions, axis=2)

    confusion_matrix(emotion, pred_emotion, sample_weight=sample_weight)

  return metrics.classification_report(confusion_matrix)

test_dataset = data_utils.load_dataset(args.test_dataset)
test_dataset = test_dataset.padded_batch(
    30,
    padded_shapes=([None, None], [None, None], [None, None]))

report = evaluate(test_dataset)

print('Evaluation Micro-f1 {:.4f} Macro-f1 {:.4f}'
    ' Weighted-f1 {:.4f} Accuracy {:.4f}'.format(
    report[1].numpy(), report[2].numpy(),
    report[3].numpy(), report[4].numpy()))
with np.printoptions(precision=4, suppress=True):
  print('Evaluation metrics of classes:\n', report[0].numpy())

