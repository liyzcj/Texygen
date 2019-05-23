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
    return total
    

def plot_data(data1, data2, mode, label1, label2):
    x1, y1 = data1[:, 0], data1[:, 1]
    x2, y2 = data2[:, 0], data2[:, 1]


    # 格式设置
    
    plt.tick_params(labelsize=12)
    if mode == 'relgan':
        plt.ylim(0.5, 1.2)
        plt.yticks(np.arange(0.5, 1.2, step=0.1))
        plt.xticks(np.arange(0, 2001, step=250))
        plt.plot([150,150], [0.505, 0.645], 'k--', linewidth=2)

    elif mode == 'seqgan':
        plt.ylim(0.7, 1.1)
        plt.yticks(np.arange(0.7, 1.1, step=0.1))
        plt.xticks(np.arange(0, 181, step=20))
        plt.plot([80,80], [0.71, 0.77], 'k--', linewidth=2)

    elif mode == 'leakgan':
        plt.ylim(0.5, 1.0)
        plt.yticks(np.arange(0.5, 1.0, step=0.1))
        plt.xticks(np.arange(0, 181, step=20))
        plt.plot([80,80], [0.6, 0.7], 'k--', linewidth=2)

    else:
        raise NotImplementedError("Error!")

    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 16,
    }

    plt.ylabel(r"$NLL_{gen}$", font2)
    plt.xlabel("Training iterations", font2)


    plt.plot(x1,y1, label=label1, linewidth=2)
    plt.plot(x2,y2, label=label2, linewidth=2)
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
    }
    plt.legend(prop=font1, loc='upper right')


if __name__ == "__main__":
    data_path = 'plotdata'
    our_seqgan = os.path.join(data_path, 'our_seqgan.csv')
    seqgan = os.path.join(data_path, 'seqgan.csv')
    our_tmp_10 = os.path.join(data_path, 'our_tmp_10.csv')
    rel_tmp_10 = os.path.join(data_path, 'rel_tmp_10.csv')
    our_tmp_50 = os.path.join(data_path, 'our_tmp_50.csv')
    rel_tmp_50 = os.path.join(data_path, 'rel_tmp_50.csv')
    leakgan = os.path.join(data_path, 'leakgan.csv')
    our_leakgan = os.path.join(data_path, 'our_leakgan.csv')
    
    leakgan = preprocess_seqgan(leakgan)
    our_leakgan = preprocess_seqgan(our_leakgan)
    our_seqgan = preprocess_seqgan(our_seqgan)
    seqgan = preprocess_seqgan(seqgan)
    our_tmp_10 = preprocess_relgan(our_tmp_10)
    rel_tmp_10 = preprocess_relgan(rel_tmp_10)
    our_tmp_50 = preprocess_relgan(our_tmp_50)
    rel_tmp_50 = preprocess_relgan(rel_tmp_50)

    plt.figure(figsize=[17,12])

    plt.subplot(221)
    plot_data(our_tmp_10, rel_tmp_10, 'relgan', "Imp-RelGan(10)", "Relgan(10)")
    plt.subplot(222)
    plot_data(our_tmp_50, rel_tmp_50, 'relgan', "Imp-RelGan(50)", "Relgan(50)")
    plt.subplot(223)
    plot_data(our_seqgan, seqgan, 'seqgan', "Imp-Seqgan", "Seqgan")
    plt.subplot(224)
    plot_data(our_leakgan, leakgan, 'leakgan', "Imp-Leakgan", "Leakgan")

    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig('figure.eps', format='eps', dpi=1000)
    plt.show()
