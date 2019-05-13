import getopt
import sys
import os

from colorama import Fore
import tensorflow as tf

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd
from models.myleakgan.Leakgan import MyLeakgan
from models.testgan.Leakgan import Testgan
from models.oldleakgan.Leakgan import OldLeakgan

from utils.config import Config

gans = {
    'seqgan': Seqgan,
    'gsgan': Gsgan,
    'textgan': TextganMmd,
    'leakgan': Leakgan,
    'rankgan': Rankgan,
    'maligan': Maligan,
    'mle': Mle,
    'myleakgan': MyLeakgan,
    'testgan': Testgan,
    'oldleakgan': OldLeakgan
}
training_mode = {'oracle', 'cfg', 'real'}

def set_gan(gan_name):
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)



def set_training(gan, training_method):
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


def def_flags():
    flags = tf.app.flags
    flags.DEFINE_enum('gan', 'mle', list(gans.keys()), 'Type of GAN to Training')
    flags.DEFINE_enum('mode', 'real', training_mode, 'Type of training mode')
    flags.DEFINE_string('data', 'data/image_coco.txt', 'Data for real Training')
    flags.DEFINE_boolean('restore', False, 'Restore models for LeakGAN')
    flags.DEFINE_string('model', "test", 'Experiment name for LeakGan')
    flags.DEFINE_integer('gpu', 0, 'The GPU used for training')
    return

def main(args):
    FLAGS = tf.app.flags.FLAGS
    gan = set_gan(FLAGS.gan)
    # experiment path
    gan.experiment_path = os.path.join('experiment', FLAGS.model)
    if not os.path.exists(gan.experiment_path):
        os.mkdir(gan.experiment_path)
    print(f"{Fore.BLUE}Experiment path: {gan.experiment_path}{Fore.RESET}")

    # tempfile
    tmp_path = os.path.join(gan.experiment_path, 'tmp')
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    gan.oracle_file = os.path.join(tmp_path, 'oracle.txt')
    gan.generator_file = os.path.join(tmp_path, 'generator.txt')
    gan.test_file = os.path.join(tmp_path, 'test_file.txt')

    # Log file
    gan.log = os.path.join(gan.experiment_path, f'experiment-log-{FLAGS.gan}-{FLAGS.mode}.csv')
    if os.path.exists(gan.log):
        print(f"{Fore.RED}[Error], Log file exist!{Fore.RESET}")
        exit(-3)

    # Config file
    config_file = os.path.join(gan.experiment_path, 'config.json')
    if not os.path.exists(config_file):
        config_file = os.path.join('models', FLAGS.gan, 'config.json')
        # copy config file
        from shutil import copyfile
        copyfile(config_file, os.path.join(gan.experiment_path, 'config.json'))
        if not os.path.exists(config_file):
            print(f"{Fore.RED}[Error], Config file not exist!{Fore.RESET}")
    print(f"{Fore.BLUE}Using config: {config_file}{Fore.RESET}")
    config = Config(config_file)
    gan.set_config(config)
    
    # output path
    gan.output_path = os.path.join(gan.experiment_path, 'output')
    if not os.path.exists(gan.output_path):
        os.mkdir(gan.output_path)

    # save path
    gan.save_path = os.path.join(gan.experiment_path, 'ckpts')
    if not os.path.exists(gan.save_path):
        os.mkdir(gan.save_path)

    train_f = set_training(gan, FLAGS.mode)
    if FLAGS.mode == 'real':
        train_f(FLAGS.data)
    else:
        train_f()

if __name__ == '__main__':
    def_flags()
    tf.app.run()