#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train.py
# Author            : Yan <yanwong@126.com>
# Date              : 15.12.2020
# Last Modified Date: 28.12.2020
# Last Modified By  : Yan <yanwong@126.com>

import time
import argparse

import numpy as np
import tensorflow as tf

import data_utils
import utterance_encoder
import emotion_model
import configuration

parser = argparse.ArgumentParser(description='Train emotion detection model.')

parser.add_argument('train_dataset', help='The path of training dataset.')
parser.add_argument('valid_dataset', help='The path of validation dataset.')
parser.add_argument('vocab', help='The vocab file.')
parser.add_argument('embeddings', help='The pre-trained word embeddings file.')
parser.add_argument('-b', '--buffer_size', type=int, default=10000,
                    help='Buffr size fo randomly shuffle the training dataset.')
parser.add_argument('checkpoints', help='The directory for saving checkpoints.')
parser.add_argument('-m', '--max_to_keep', type=int, default=50,
                    help='The maximum checkpoints to keep.')

args = parser.parse_args()

train_dataset = data_utils.load_dataset(args.train_dataset)
val_dataset = data_utils.load_dataset(args.valid_dataset)

model_config = configuration.ModelConfig()
train_config = configuration.TrainingConfig()

train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(args.buffer_size)
train_dataset = train_dataset.padded_batch(
    train_config.batch_size,
    padded_shapes=([None, None], [None, None], [None, None]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_dataset.padded_batch(
    train_config.batch_size,
    padded_shapes=([None, None], [None, None], [None, None]))

vocab = data_utils.load_vocab(args.vocab)
w2v = data_utils.build_w2v(args.embeddings, vocab, model_config.d_word)
embeddings = data_utils.build_vocab_embeddings(w2v, vocab, model_config.d_word)


utter_encoder = utterance_encoder.CnnUtteranceEncoder(
    len(vocab),
    model_config.filters,
    model_config.kernel_sizes,
    model_config.d_word,
    model_config.d_sent)
model = emotion_model.ContextFreeModel(utter_encoder, model_config.n_classes)

optimizer = tf.keras.optimizers.Adam(learning_rate=train_config.learning_rate)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoints, args.max_to_keep)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def loss_function(real, pred):
  # real.shape == (batch_size, dial_len)
  # pred.shape == (batch_size, dial_len, n_classes)

  loss = loss_object(real, pred)  # (batch_size, dial_len)
  mask = tf.cast(tf.math.not_equal(real, 0), dtype=loss.dtype)
  loss *= mask
  return tf.math.reduce_mean(loss)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

def train_step(speaker, utterance, emotion):
  # speaker.shape == (batch_size, 1, dial_len)
  # emotion.shape == (batch_size, 1, dial_len)
  # utterance.shape == (batch_size, dial_len, sent_len)

  speaker = tf.squeeze(speaker)  # (batch_size, dial_len)
  emotion = tf.squeeze(emotion)  # (batch_size, dial_len)

  mask = tf.cast(tf.math.not_equal(utterance, 0), dtype=tf.float32)

  with tf.GradientTape() as tape:
    predictions = model(utterance, True, mask)  # (batch_size, dial_len, n_classes)
    loss = loss_function(emotion, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(emotion, tf.math.argmax(predictions, axis=2))

for epoch in range(train_config.n_epochs):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  for (batch, (speaker, utterance, emotion)) in enumerate(train_dataset):
    train_step(speaker, utterance, emotion)

    if batch % 50 == 0:
      print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                       train_loss.result(),
                                                       train_accuracy.result()))
  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

# utter_encoder = utterance_encoder.CnnUtteranceEncoder(3000, 50, [3, 4, 5], 300, 300)
# # model = emotion_model.ContextFreeModel(utter_encoder, 6)
# model = emotion_model.BiLstmModel(utter_encoder, 100, 0.1, 6)
# x = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], [[1, 3, 5, 7, 9], [2, 4, 6, 8, 0]]])
# mask = np.array([[[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]])
# pred = model(x, True, mask)
# 
# print(pred.shape)

