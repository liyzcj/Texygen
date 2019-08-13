import os
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
import csv
sys.path.append('d:\\Texygen')

def preprocess_relgan(datafile):
    total = []
    with open(datafile) as inf:
        data = csv.reader(inf)
        for i in data:
            if data.line_num == 1:
                continue
            total.append(i)
    total = np.array(total, dtype=np.float)
    res = np.stack([total[:,0], total[:,2]], axis=1)
    return res


def preprocess_seqgan(datafile):
    total = []
    with open(datafile) as inf:
        data = csv.reader(inf)
        for i in data:
            if data.line_num == 1:
                continue
            total.append(i)
    total = np.array(total, dtype=np.float)
    res = np.zeros((11, 5))
    for i in range(11):
        res[i] = (total[i * 3] + total[i * 3 + 1] + total[i * 3 + 2])/3
    return res
    

def plot_data(data, label):
    x1, y1, y2, y3, y4 = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


    # 格式设置
    
    plt.tick_params(labelsize=12)

    plt.ylim(0, 1.0)
    plt.xlim(100, 1300)
    plt.yticks(np.arange(0.0, 1.0, step=0.1))
    plt.xticks(np.arange(100, 1300, step=100))


    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 16,
    }

    plt.ylabel(label, font2)
    plt.xlabel("Number of samples", font2)


    plt.plot(x1,y1, 'o-' ,label='BLEU-2',  linewidth=2)
    plt.plot(x1,y2, '<-', label='BLEU-3',  linewidth=2)
    plt.plot(x1,y3, ',-', label='BLEU-4',  linewidth=2)
    plt.plot(x1,y4, '.-', label='BLEU-5',  linewidth=2)
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 14,
    }
    plt.legend(prop=font1, loc='upper right')


if __name__ == "__main__":
    data_path = 'utils/others'
    bleu = os.path.join(data_path, 'bleu_result.csv')
    self_bleu = os.path.join(data_path, 'selfbleu_result.csv')
    
    bleu = preprocess_seqgan(bleu)
    self_bleu = preprocess_seqgan(self_bleu)

    plt.figure(figsize=[17,7])

    plt.subplot(121)
    plot_data(bleu, 'BLEU scores')
    plt.subplot(122)
    plot_data(self_bleu, 'Self-BLEU scores')

    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('figure.eps', format='eps', dpi=300)
    plt.show()
