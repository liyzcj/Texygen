"""
Computing the self-bleu of training file.
"""

import sys
sys.path.append('d:\\Texygen')
from utils.metrics.SelfBleu import SelfBleu
from time import time
import os
import json

train_file = os.path.join('data', 'image_coco.txt')
test_file = os.path.join('data', 'testdata', 'test_coco.txt')

if __name__ == "__main__":
    
  score = {}
  samples = [1700, 1800, 1900, 2000]
  exp_num = 3
  log_file = open('selfbleu_result.csv', 'a')

  bleu = SelfBleu(train_file)

  # head = "samples, sBleu-2, sBleu-3, sBleu-4, sBleu-5\n"
  # log_file.write(head)
  for nsam in samples:
    bleu.sample_size = nsam
    for n_exp in range(exp_num):
      scores = [nsam]
      for i in range(2,6):
        print(f"Samples:{nsam}, Exp Number:{n_exp}, Gram: {i}", end='')
        bleu.gram = i
        tic = time()
        scores.append(bleu.get_score())
        toc = time()
        print(f"\t Time: {toc - tic:.1f}s")
      buffer = ','.join([str(s) for s in scores])+'\n'
      log_file.write(buffer)
  log_file.close()