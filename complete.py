import argparse
import os
import tensorflow as tf

from model import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--approach', type=str,
                    choices=['adam', 'hmc'],
                    default='adam')
parser.add_argument('--lr', type=float, default=0.004)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hmcBeta', type=float, default=0.2)
parser.add_argument('--hmcEps', type=float, default=0.001)
parser.add_argument('--hmcL', type=int, default=100)
parser.add_argument('--hmcAnneal', type=float, default=1)
parser.add_argument('--nIter', type=int, default=1500)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.05)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('--outInterval', type=int, default=50)
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full', 'grid', 'lowres', 'crop', 'eye'],
                    default='center')
parser.add_argument('--cropScale', type=float, default=0.85)
parser.add_argument('--centerScale', type=float, default=0.25)
parser.add_argument('--imgs', type=str, nargs='+', default=['./data/celebA/*'])
parser.add_argument('--num', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset', type=str, choices=['mnist', 'celebA'], default='celebA')

args = parser.parse_args()

assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if args.dataset == 'celebA':
    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess, checkpoint_dir=args.checkpointDir, lam=args.lam, batch_complete_size=args.batch_size, batch_size=args.batch_size, input_height=108, input_width=108)
        dcgan.complete(args)
else:
    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess, checkpoint_dir=args.checkpointDir, lam=args.lam, batch_complete_size=args.batch_size, batch_size=args.batch_size, input_height=28, input_width=28, output_height=28, output_width=28, dataset_name='mnist', y_dim=10)
        dcgan.complete(args)
