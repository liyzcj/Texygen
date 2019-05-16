from abc import abstractmethod, ABCMeta

from utils.utils import init_sess


class Gan(metaclass=ABCMeta):

    def __init__(self):
        self.oracle = None
        self.generator = None
        self.discriminator = None
        self.gen_data_loader = None
        self.dis_data_loader = None
        self.oracle_data_loader = None
        self.sess = init_sess()
        self.metrics = list()
        self.epoch = 0
        self.log = None
        self.reward = None
        # temp file
        self.oracle_file = None
        self.generator_file = None
        self.test_file = None
        # pathes
        self.output_path = None
        self.save_path = None
        self.summary_path = None
        # dict
        self.wi_dict = None
        self.iw_dict = None

    def set_config(self, config):
        self.__dict__.update(config.dict)

    def set_oracle(self, oracle):
        self.oracle = oracle

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_data_loader(self, gen_loader, dis_loader, oracle_loader):
        self.gen_data_loader = gen_loader
        self.dis_data_loader = dis_loader
        self.oracle_data_loader = oracle_loader

    def set_sess(self, sess):
        self.sess = sess

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_epoch(self):
        self.epoch += 1

    def reset_epoch(self):
        # in use
        self.epoch = 0
        return

    def evaluate(self):
        from time import time
        log = "epoch:" + str(self.epoch) + '\t'
        scores = list()
        scores.append(self.epoch)
        for metric in self.metrics:
            tic = time()
            score = metric.get_score()
            log += metric.get_name() + ":" + str(score) + '\t'
            toc = time()
            print(f"time elapsed of {metric.get_name()}: {toc - tic:.1f}s")
            scores.append(score)
        print(log)
        return scores

    def check_valid(self):
        # TODO
        pass

    @abstractmethod
    def train_oracle(self):
        pass

    def train_cfg(self):
        pass

    def train_real(self):
        pass


class Gen(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def get_nll(self):
        pass

    @abstractmethod
    def pretrain_step(self):
        pass


class Dis(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        pass
