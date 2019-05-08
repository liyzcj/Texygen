import getopt
import sys

from colorama import Fore
import tensorflow as tf

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd


gans = {
    'seqgan': Seqgan,
    'gsgan': Gsgan,
    'textgan': TextganMmd,
    'leakgan': Leakgan,
    'rankgan': Rankgan,
    'maligan': Maligan,
    'mle': Mle
}
training_mode = {'oracle', 'cfg', 'real'}

def set_gan(gan_name):
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 5000
        gan.generate_num = 10000
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
    flags.DEFINE_enum('mode', 'oracle', training_mode, 'Type of training mode')
    flags.DEFINE_string('data', 'data/image_coco.txt', 'Data for real Training')
    flags.DEFINE_boolean('restore', False, 'Restore models for LeakGAN')
    flags.DEFINE_boolean('resD', False, 'Restore discriminator for LeakGAN')
    flags.DEFINE_integer('length', 20, 'Sequence Length for LeakGAN oracle training')
    flags.DEFINE_string('model', "test", 'Experiment name for LeakGan')
    flags.DEFINE_integer('gpu', 0, 'The GPU used for training')
    return

def main(args):
    FLAGS = tf.app.flags.FLAGS
    gan = set_gan(FLAGS.gan)
    train_f = set_training(gan, FLAGS.mode)
    if FLAGS.mode == 'real':
        train_f(FLAGS.data)
    else:
        train_f()

if __name__ == '__main__':
    def_flags()
    tf.app.run()