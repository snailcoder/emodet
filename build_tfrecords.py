#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : build_tfrecords.py
# Author            : Yan <yanwong@126.com>
# Date              : 07.04.2020
# Last Modified Date: 29.12.2020
# Last Modified By  : Yan <yanwong@126.com>

import os
import collections
import argparse
import logging

import numpy as np
import tensorflow as tf

import special_words
import data_utils

logging.basicConfig(level=logging.INFO)

def _build_vocab(input_file, output_dir):
  """ Build a vocab according to the corpus.

  Args:
    input_file: The processed corpus. Each line contains three fields:
      speaker id, utterance, and emotion label, which are separated by tab.

  Returns:
    An ordered dict mapping each character to its Id.
  """
  word_cnt = collections.Counter()

  with tf.io.gfile.GFile(input_file, mode='r') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      _, u, _ = line.split('\t')
      u = data_utils.clean_str(u)
      word_cnt.update(u.split())

  sorted_items = word_cnt.most_common()

  vocab = collections.OrderedDict()
  vocab[special_words.PAD] = special_words.PAD_ID
  vocab[special_words.UNK] = special_words.UNK_ID

  for i, item in enumerate(sorted_items):
    vocab[item[0]] = i + 2  # 0: PAD, 1: UNK
  
  logging.info('Create vocab with %d words.', len(vocab))

  vocab_file = os.path.join(output_dir, 'vocab.txt')
  with tf.io.gfile.GFile(vocab_file, mode='w') as f:
    f.write('\n'.join(vocab.keys()))

  logging.info('Wrote vocab file to %s', vocab_file)

  return vocab

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(
      int64_list=tf.train.Int64List(value=[int(v) for v in value]))

def _utterance_to_ids(utter, vocab):
  """Helper for converting an utterance (string) to a list of ids."""
  utter = data_utils.clean_str(utter)
  utter = utter.split()
  ids = [vocab.get(w, special_words.UNK_ID) for w in utter]
  return ids

def _create_serialized_example(speakers, utterances, emotions, vocab):
  """Helper for creating a serialized Example proto."""
  example = tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list={
    'utterances': tf.train.FeatureList(
      feature=[_int64_feature(_utterance_to_ids(u, vocab)) for u in utterances]),
    'speakers': tf.train.FeatureList(
      feature=[_int64_feature(speakers)]),
    'emotions': tf.train.FeatureList(
      feature=[_int64_feature(emotions)])}))
  return example.SerializeToString()

def _build_IEMOCAP_dataset(filename, vocab):
  """ Build dataset from the file containing dialogues of IEMOCAP.

  Args:
    filename: The file containing dialogues which are separated by empty lines.
    vocab: A dict mapping each word to Id.

  Returns:
    A list containing serialized examples.

  """
  serialized = []
  speaker_map = {'F': 0, 'M': 1}
  emotion_map = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}

  with tf.io.gfile.GFile(filename, 'r') as f:
    speakers = []
    utterances = []
    emotions = []

    for line in f:
      line = line.strip()
      if not line:
        serialized.append(
            _create_serialized_example(speakers, utterances, emotions, vocab))
        speakers = []
        utterances = []
        emotions = []
      else:
        fields = line.split('\t')
        assert(len(fields) == 3)
        speakers.append(speaker_map.get(fields[0].split('_')[-1].strip('0123456789')))
        utterances.append(fields[1])
        emotions.append(emotion_map[fields[2]])

  return serialized

def _write_shard(filename, dataset, indices):
  """Writes a TFRecord shard."""
  with tf.io.TFRecordWriter(filename) as writer:
    for j in indices:
      writer.write(dataset[j])

def _write_dataset(name, dataset, indices, num_shards, output_dir):
  """Writes a sharded TFRecord dataset.

  Args:
    name: Name of the dataset (e.g. "train").
    dataset: List of serialized Example protos.
    indices: List of indices of 'dataset' to be written.
    num_shards: The number of output shards.
  """
  borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
  for i in range(num_shards):
    filename = os.path.join(
        output_dir, '%s-%.5d-of-%.5d' % (name, i, num_shards))
    shard_indices = indices[borders[i]:borders[i + 1]]
    _write_shard(filename, dataset, shard_indices)
    logging.info('Wrote dataset indices [%d, %d) to output shard %s',
                 borders[i], borders[i + 1], filename)

def main():
  parser = argparse.ArgumentParser(
      description='Make processed corpus datasets.')

  parser.add_argument(
      'input_file',
      help='The processed corpus of which each line contains speaker id,'
           ' utterance and emotion label.')
  parser.add_argument('output_dir', help='The output directory.')
  parser.add_argument(
      '-validation_percentage', type=float, default=0.1,
      help='Percentage of the training data used for validation.')
  parser.add_argument('-train_shards', type=int, default=100,
                      help='Number of output shards for the training set.')
  parser.add_argument('-validation_shards', type=int, default=1,
                      help='Number of output shards for the validation set.')

  args = parser.parse_args()

  if not tf.io.gfile.isdir(args.output_dir):
    tf.io.gfile.makedirs(args.output_dir)

  vocab = _build_vocab(args.input_file, args.output_dir)
  dataset = _build_IEMOCAP_dataset(args.input_file, vocab)

  logging.info('Shuffling dataset.')
  np.random.seed(123)
  shuffled_indices = np.random.permutation(len(dataset))
  num_validation_sentences = int(args.validation_percentage * len(dataset))

  val_indices = shuffled_indices[:num_validation_sentences]
  train_indices = shuffled_indices[num_validation_sentences:]

  if args.validation_percentage < 1.:
    _write_dataset('train', dataset, train_indices,
                   args.train_shards, args.output_dir)
  if args.validation_percentage > 0.:
    _write_dataset('valid', dataset, val_indices,
                   args.validation_shards, args.output_dir)

if __name__ == '__main__':
  main()

