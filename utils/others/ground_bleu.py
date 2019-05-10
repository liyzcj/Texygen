import sys
sys.path.append('d:\\Texygen')
from utils.metrics.Bleu import Bleu
import os
import json

train_file = os.path.join('data', 'image_coco.txt')
test_file = os.path.join('data', 'testdata', 'test_coco.txt')

if __name__ == "__main__":
    
  score = {}

  bleu = Bleu(train_file, test_file)

  for i in range(2,6):
    bleu.gram = i
    score[f"Bleu{i}"] = bleu.get_score()

  print(score)

  with open('res.json', 'w') as f:
    json.dump(score, f)

## Result
# {
#   "Bleu2": 0.5176190691145264, 
#   "Bleu3": 0.2837650014127829, 
#   "Bleu4": 0.16248278209254763, 
#   "Bleu5": 0.10303549181525805
# }