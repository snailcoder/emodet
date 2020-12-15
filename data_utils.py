#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_utils.py
# Author            : Yan <yanwong@126.com>
# Date              : 03.12.2020
# Last Modified Date: 15.12.2020
# Last Modified By  : Yan <yanwong@126.com>

import logging
import glob
import re

import tensorflow as tf

def clean_str(string):
  """
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\.]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\.", " \.", string)
  string = re.sub(r"( \.){3}", " (\.){3}", string)
  string = re.sub(r"\s{2,}", " ", string)   
  return string.strip().lower()

def create_dataset(file_pattern):
  """Fetches examples from disk into tf.data.TFRecordDataset.

    Args:
      file_pattern: Comma-separated list of file patterns (e.g.
        "/tmp/train_data-?????-of-00100", where '?'
        acts as a wildcard that matches any character).

    Returns:
      A dataset read from TFRecord files.
  """

  data_files = []
  for pattern in file_pattern.split(','):
    data_files.extend(glob.glob(pattern))
  if not data_files:
    logging.fatal('Found no input files matching %s', file_pattern)
  else:
    logging.info('Prefetching values from %d files matching %s',
                  len(data_files), file_pattern)

  dataset = tf.data.TFRecordDataset(data_files)

  def _parse_record(record):
    features = {
      'utterances': tf.io.VarLenFeature(dtype=tf.int64),
      'speakers': tf.io.VarLenFeature(dtype=tf.int64),
      'emotions': tf.io.VarLenFeature(dtype=tf.int64)
    }
    parsed_features = tf.io.parse_single_sequence_example(
        record, sequence_features=features)

    utterances = tf.sparse.to_dense(parsed_features[1]['utterances'])
    speakers = tf.sparse.to_dense(parsed_features[1]['speakers'])
    emotions = tf.sparse.to_dense(parsed_features[1]['emotions'])

    return speakers, utterances, emotions

  dataset = dataset.map(_parse_record)

  return dataset

