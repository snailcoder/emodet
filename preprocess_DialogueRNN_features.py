#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : preprocess_DialogueRNN_features.py
# Author            : Yan <yanwong@126.com>
# Date              : 08.01.2021
# Last Modified Date: 09.01.2021
# Last Modified By  : Yan <yanwong@126.com>

import argparse
import logging
import pickle
import random
import os

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(
      float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(
      int64_list=tf.train.Int64List(value=[int(v) for v in value]))

def _create_serialized_example(speakers, utterances, emotions):
  """Helper for creating a serialized Example proto."""
  example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list={
    'utterances': tf.train.FeatureList(
      feature=[_float_feature(u) for u in utterances]),
    'speakers': tf.train.FeatureList(
      feature=[_int64_feature(speakers)]),
    'emotions': tf.train.FeatureList(
      feature=[_int64_feature(emotions)])}))
  return example.SerializeToString()

def _build_IEMOCAP_features_dataset(filename, val_rate):
  video_ids, video_speakers, video_labels, video_text, \
  video_audio, video_visual, video_sentence, trainVids, \
  test_vids = pickle.load(open(filename, 'rb'), encoding='latin1')

  val_size = int(len(trainVids) * val_rate)
  trainVids = list(trainVids)
  train_vids, val_vids = trainVids[val_size:], trainVids[:val_size]
  # random.shuffle(train_vids)
  # random.shuffle(val_vids)
  # random.shuffle(test_vids)
  
  speaker_map = {'F': 0, 'M': 1}

  train, val, test = [], [], []

  for vid in train_vids:
    serialized = _create_serialized_example(
        [speaker_map[s] for s in video_speakers[vid]],
        video_text[vid],
        video_labels[vid])
    train.append(serialized)

  for vid in val_vids:
    serialized = _create_serialized_example(
        [speaker_map[s] for s in video_speakers[vid]],
        video_text[vid],
        video_labels[vid])
    val.append(serialized)

  for vid in test_vids:
    serialized = _create_serialized_example(
        [speaker_map[s] for s in video_speakers[vid]],
        video_text[vid],
        video_labels[vid])
    test.append(serialized)

  return train, val, test

def _write_shard(filename, dataset, indices):
  """Writes a TFRecord shard."""
  with tf.io.TFRecordWriter(filename) as writer:
    for j in indices:
      writer.write(dataset[j])

def _write_dataset(name, dataset, num_shards, output_dir):
  """Writes a sharded TFRecord dataset.

  Args:
    name: Name of the dataset (e.g. "train").
    dataset: List of serialized Example protos.
    num_shards: The number of output shards.
  """
  borders = np.int32(np.linspace(0, len(dataset), num_shards + 1))
  indices = list(range(len(dataset)))

  for i in range(num_shards):
    filename = os.path.join(
        output_dir, '%s-%.5d-of-%.5d' % (name, i, num_shards))
    shard_indices = indices[borders[i]:borders[i + 1]]
    _write_shard(filename, dataset, shard_indices)
    logging.info('Wrote dataset indices [%d, %d) to output shard %s',
                 borders[i], borders[i + 1], filename)

def main():
  parser = argparse.ArgumentParser(
      description='Processed pickled features provided by DialogueRNN authors.')

  parser.add_argument(
      'input_file',
      help='The pickle file provided by DialogueRNN authors.')
  parser.add_argument('output_dir', help='The output directory.')
  parser.add_argument(
      '-validation_percentage', type=float, default=0.1,
      help='Percentage of the training data used for validation.')
  parser.add_argument('-train_shards', type=int, default=100,
                      help='Number of output shards for the training set.')
  parser.add_argument('-validation_shards', type=int, default=1,
                      help='Number of output shards for the validation set.')
  parser.add_argument('-test_shards', type=int, default=1,
                      help='Number of output shards for the test set.')

  args = parser.parse_args()

  if not tf.io.gfile.isdir(args.output_dir):
    tf.io.gfile.makedirs(args.output_dir)

  train, val, test = _build_IEMOCAP_features_dataset(args.input_file,
                                                     args.validation_percentage)

  _write_dataset('train', train, args.train_shards, args.output_dir)
  _write_dataset('valid', val, args.validation_shards, args.output_dir)
  _write_dataset('test', test, args.test_shards, args.output_dir)

if __name__ == '__main__':
  main()

