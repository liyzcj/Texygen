"""
test the bleu value of a file
"""

import sys
sys.path.append('d:\\Texygen')
# from utils.metrics.Bleu import Bleu
from utils.others.Bleu import Bleu
import os
import json
from time import time


GRAM = [2,3,4,5]
filt_to_test = sys.argv[-1]
test_file = os.path.join('data', 'testdata', 'test_coco.txt')

bleu = Bleu(filt_to_test, test_file)
for i in GRAM:
    bleu.gram = i
    bleu.set_name(f"Bleu{i}")
    tic = time()
    score = bleu.get_score()
    toc = time()
    print(f"{bleu.name} Score: {score}   Time: {toc-tic:.1f}s")
    # 0.5310657

    # 170epoch 0.5152922818225858
    # 2000:    0.5459362050313327
    # 2051:    0.528616088847338