#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_utils.py
# Author            : Yan <yanwong@126.com>
# Date              : 03.12.2020
# Last Modified Date: 16.12.2020
# Last Modified By  : Yan <yanwong@126.com>

import logging
import glob
import re
import collections

import numpy as np
import tensorflow as tf

def load_vocab(vocab_file):
  """ Load vocab as an ordered dict.

  Args:
    vocab_file: The vocab file in which each line is a single word.

  Returns:
    An ordered dict of which key is the word and value is id.

  """
  vocab = collections.OrderedDict()

  with tf.io.gfile.GFile(vocab_file, 'r') as f:
    wid = 0
    for line in f:
      w = line.strip()
      vocab[w] = wid
      wid += 1
    logging.info('# Total words in vocab: %d', wid)
  return vocab

def build_w2v(fname, vocab):
  """ Build embedding lookup table for words in vocab.

  Args:
    fname: The pre-trained word vector file, e.g. GloVe.
    vocab: A dict of words appearing in training corpus.

  Returns:
    A dict of pre-trained word vectors for words in vocab.
  """
  w2v = {}
  with tf.io.gfile.GFile(fname, 'r') as f:
    for line in f:
      fields = line.split()
      w, v = fields[0], fields[1:]
      if w in vocab:
        w2v[w] = np.array(v).astype(np.float)

  logging.info('%d words in vocab are assigned pretrained embeddings', len(w2v))

  return w2v

def build_vocab_embeddings(w2v, vocab, d_word):
  """Assign word embedding for each word in vocab. For the word that's
     in vocab but there's no corresponding pre-trained embedding, generate a
     embedding randomly for it.

  Args:
    w2v: A dict contains pre-trained word embedding. Each word in this
      dict is also in the vocab.
    vocab: An ordered dict of which key is the word and value is id.
    d_word: The dimension of word embeddings.

  Returns:
    A word embedding list contains all words in vocab.
  """
  embeddings = []
  for word in vocab:
    emb = w2v.get(word, None)
    if emb is None:
      logging.warning('Word not in vocab: %s', word)
      emb = np.random.uniform(-0.25, 0.25, d_word)
    embeddings.append(emb)

  return np.array(embeddings)

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

