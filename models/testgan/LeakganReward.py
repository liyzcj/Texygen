import numpy as np


def redistribution(idx, total, min_v):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))


def rescale(reward, rollout_num=1.0):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l), min_s)
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
    return ret



class Reward(object):
    def __init__(self, dis, sess):
        self.dis = dis
        self.sess = sess

    def get_reward(self, input_x):
        feed = {self.dis.D_input_x: input_x}
        ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
        ypred = np.array([item[1] for item in ypred_for_auc])
        batch_size = ypred.shape[0]
        rewards = np.transpose(ypred.reshape([1, batch_size]))
        return rewards