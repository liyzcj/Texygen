"""
test the bleu value of a file
"""

import sys
sys.path.append('d:\\Texygen')
# from utils.metrics.Bleu import Bleu
from models.mrelgan.Bleu import Bleu
import os
import json
from time import time

GRAM = 3
filt_to_test = 'epoch_140_2051.txt'
test_file = os.path.join('data', 'testdata', 'test_coco.txt')

if __name__ == "__main__":
    bleu = Bleu(filt_to_test, test_file, gram=GRAM)
    bleu.set_name(f"Bleu{GRAM}")
    tic = time()
    score = bleu.get_score()
    toc = time()

    print(f"{bleu.name} Score: {score}   Time: {toc-tic:.1f}s")
    # 0.5310657

    # 170epoch 0.5152922818225858
    # 2000:    0.5459362050313327
    # 2051:    0.528616088847338