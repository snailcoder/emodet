#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : preprocess_dataset.py
# Author            : Yan <yanwong@126.com>
# Date              : 05.12.2020
# Last Modified Date: 08.12.2020
# Last Modified By  : Yan <yanwong@126.com>

import re
import collections
import os
import argparse

class UtteranceExample:
  def __init__(self, utterance=None, speaker=None, emotion=None):
    self.utterance = utterance
    self.speaker = speaker
    self.emotion = emotion

def combine_IEMOCAP_script_and_emotion(script_file, emotion_file):
  """ For IEMOCAP dataset, assign labels given by the evaluation 
      file to corresponding utterances in the transcription file.

  Args:
    script_file: The file containing utterances consisting of a session.
    emotion_file: The file containing corresponding emotion annotations of
      the script_file
  """

  umap = collections.OrderedDict()

  with open(script_file, 'r') as f:
    for line in f:
      fields = re.split("\[.+-.+\]:", line)
      if len(fields) != 2:
        print('Invalid line in ' + script_file + ": " + line)
        continue
      uk, u = fields
      # s = uk.split('_')[2].strip(' 0123456789')
      # umap[uk.strip()] = UtteranceExample(utterance=u.strip(), speaker=s)
      umap[uk.strip()] = UtteranceExample(utterance=u.strip(), speaker=uk.strip())

  with open(emotion_file, 'r') as f:
    for line in f:
      if re.match('^\[(\d+\.\d+) - (\d+\.\d+)\]', line):
        _, uk, e, _ = line.split('\t')
        assert(uk in umap)
        umap[uk].emotion = e

  return umap.values()

def process_IEMOCAP(session_path, dialogue_file, emoset):
  """ Merge some IEMOCAP dialogues to a single file.

  Args:
    session_path: iemocap/s1, or iemocap/s2,..., or iemocap/s5
    dialogue_file: Save all dialogues in session_path to a single file.
    emoset: Only emotions in this set should appear in the final result.
  """
  script_path = os.path.join(session_path, 'transcriptions')
  emotion_path = os.path.join(session_path, 'EmoEvaluation')
  filenames = os.listdir(script_path)

  n_utterances = 0
  n_confused = 0
  n_dialogues = 0

  with open(dialogue_file, 'w') as fo:
    for name in filenames:
      script_file = os.path.join(script_path, name)
      emotion_file = os.path.join(emotion_path, name)
      dialogue = combine_IEMOCAP_script_and_emotion(script_file, emotion_file)

      for example in dialogue:
        if example.emotion in emoset:
          fo.write('%s\t%s\t%s\n' %
              (example.speaker, example.utterance, example.emotion))
          n_utterances += 1
        else:
          n_confused += 1

      fo.write('\n')
      n_dialogues += 1
    print('# Dialogues: %d\t# Utterances: %d\t# Confused: %d'
        % (n_dialogues, n_utterances, n_confused))

def main():
  parser = argparse.ArgumentParser(
      description='Convert original datasets to a convenient '
                  'data format for following processing')
  parser.add_argument('dataset', choices=['iemocap'],
                      help='The name of dataset for preprocessing.')
  parser.add_argument('session_path',
                      help='The lowest-level directory of the original dataset'
                           ' containing whole dialogue utterances, speakers and'
                           ' emotions. For IEMOCAP, it is iemocap/s1, iemocap/s2'
                           ' etc.')
  parser.add_argument('-e', '--emotions', nargs='+',
                      help='Emotions appear in the output file. For IEMOCAP,'
                           ' emotions is a subset of {ang, exc, fea, fru, hap,'
                           ' neu, sad, sur, dis}.')
  parser.add_argument('output',
                      help='The result file. All dialogues are merged into one'
                           ' file, dialogue examples are separated by empty lines.')
  args = parser.parse_args()

  if args.dataset == 'iemocap':
    process_IEMOCAP(args.session_path, args.output, set(args.emotions))

if __name__ == '__main__':
  main()

