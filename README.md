# emodet
Implement state-of-the-art models for emotion detection in conversation.

# Datasets
For the original IEMOCAP dataset, utterances are annotated with categorical labels, which are among angry, disgust, fear, frustrated, sad, neutral, excited, happy, surprised, other and XXX (when annotators were not able to have agreement on the label). For example, if we only retain utterances with emotions of angry, frustrated, sad, neutral, excited, happy, just like what some works on emotion detection such as [DialogueRNN](https://arxiv.org/pdf/1811.00405.pdf) does, the statistics of each session(s1-s5) are as follows:

| session id | # utterances |
|  :----:    |    :----:    |
|    s1      |     1365     |
|    s2      |     1348     |
|    s3      |     1533     |
|    s4      |     1512     |
|    s5      |     1622     |

Previous works modified some XXXs to meanningful labels for the need of a better research. However, they don't provide details for this modification. Hence, I conducted a reverse analysis on some given features data and found out following XXX labels are all changed to "hap":

| session id | utterance id |
| :--------: |     :----    |
|     s1     |  Ses01F_impro03_F007, Ses01F_impro03_F011, Ses01F_impro03_F014, Ses01F_impro03_M012, Ses01F_impro06_M017, Ses01F_impro07_F007, Ses01M_impro03_F000, Ses01M_impro07_F002   |
|     s2     |  Ses02F_impro03_F002, Ses02F_impro03_F005, Ses02F_impro03_F006, Ses02F_impro03_F007, Ses02F_impro03_F020, Ses02M_impro03_F010, Ses02M_impro03_F026, Ses02M_impro03_F027   |
|     s3     |  Ses03F_impro02_M003, Ses03F_impro03_F000, Ses03F_impro03_F003, Ses03F_impro03_F010, Ses03F_impro03_M003, Ses03F_impro03_M008, Ses03F_impro03_M013, Ses03F_impro03_M016, Ses03F_impro03_M019, Ses03F_impro04_M020, Ses03F_impro07_F000, Ses03F_impro07_F002, Ses03F_impro07_F006, Ses03F_impro07_M016, Ses03F_impro07_M020, Ses03F_impro07_M031, Ses03M_impro02_F001, Ses03M_impro03_F000, Ses03M_impro03_F019, Ses03M_impro03_M005, Ses03M_impro03_M006, Ses03M_impro03_M008, Ses03M_impro03_M009, Ses03M_impro03_M010, Ses03M_impro03_M024, Ses03M_impro03_M025, Ses03M_impro03_M026, Ses03M_impro03_M027, Ses03M_impro03_M034, Ses03M_impro03_M036, Ses03M_impro03_M037, Ses03M_impro05a_F017, Ses03M_impro07_F020, Ses03M_impro07_M001, Ses03M_impro07_M012, Ses03M_impro07_M021  |
|     s5     |  Ses05F_impro07_M010  |

These changes lead to a new statistics which is the same with that [DialogueRNN](https://arxiv.org/pdf/1811.00405.pdf) reported:

| session id | # utterances |
|  :----:    |    :----:    |
|    s1      |     1372     |
|    s2      |     1356     |
|    s3      |     1569     |
|    s4      |     1512     |
|    s5      |     1623     |
