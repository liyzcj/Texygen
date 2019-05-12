"""
Computing the bleu of training file by test file
"""

import sys
sys.path.append('d:\\Texygen')
from utils.metrics.Bleu import Bleu
import os
import json
from time import time

train_file = os.path.join('data', 'image_coco.txt')
test_file = os.path.join('data', 'testdata', 'test_coco.txt')

if __name__ == "__main__":
    
  samples = [900, 1000, 1100, 1200]
  exp_num = 3
  log_file = open('bleu_result.csv', 'a')

  bleu = Bleu(train_file, test_file)

  # head = "samples, Bleu-2, Bleu-3, Bleu-4, Bleu-5\n"
  # log_file.write(head)
  for nsam in samples:
    bleu.sample_size = nsam
    for n_exp in range(exp_num):
      scores = [nsam]
      for i in range(2,6):
        print(f"Samples:{nsam}, Exp Number:{n_exp}, Gram: {i}")
        bleu.gram = i
        tic = time()
        scores.append(bleu.get_score())
        toc = time()
        print(f"\t Time: {toc - tic:.1f}s")
      buffer = ','.join([str(s) for s in scores])+'\n'
      log_file.write(buffer)
  log_file.close()


## Result
# {
#   "Bleu2": 0.5176190691145264, 
#   "Bleu3": 0.2837650014127829, 
#   "Bleu4": 0.16248278209254763, 
#   "Bleu5": 0.10303549181525805
# }