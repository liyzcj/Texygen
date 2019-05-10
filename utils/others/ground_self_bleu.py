"""
Computing the self-bleu of training file.
"""

import sys
sys.path.append('d:\\Texygen')
from utils.metrics.SelfBleu import SelfBleu
import os
import json

train_file = os.path.join('data', 'image_coco.txt')
test_file = os.path.join('data', 'testdata', 'test_coco.txt')

if __name__ == "__main__":
    
  score = {}

  bleu = SelfBleu(train_file)

  for i in range(2,6):
    bleu.gram = i
    score[f"Bleu{i}"] = bleu.get_score(is_fast=False)

  print(score)

  with open('res.json', 'w') as f:
    json.dump(score, f)