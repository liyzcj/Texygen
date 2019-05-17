import sys
import os
from time import time
sys.path.append('d:\\Texygen')
from utils.others.Bleu import Bleu


test_folder = sys.argv[-1]
all_file = os.listdir(test_folder)
test = os.path.join('data', 'testdata', 'test_coco.txt')
GRAM = 3
log_file = os.path.join(test_folder, 'eval_log.csv')
if not os.path.exists(test_folder):
    print("Error, no such folder!")
with open(log_file, 'w')  as log:
    log.write('epoch, bleu3\n')

for test_file in all_file:
    epoch = test_file.split('.')[0].split('_')[-1]
    test_file = os.path.join(test_folder, test_file)
    bleu = Bleu(test_file, test, gram=GRAM)
    bleu.set_name(f"Bleu{GRAM}")
    tic = time()
    score = bleu.get_score()
    toc = time()

    print("Epoch {}, {} Score: {}  Time: {:.1f}".format(
        epoch, bleu.name, score, toc-tic
    ))
    with open(log_file, 'a') as log:
        log.write('{}, {}\n'.format(epoch, score))