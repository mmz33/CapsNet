from engine import Engine
from dataset import load_mnist
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test',  action='store_true', help='Run testing')
    args = parser.parse_args(sys.argv[1:])

    datasets = load_mnist()
    engine = Engine(datasets)
    if args.train:
        engine.init_engine()
        engine.train(restore_checkpoint=True)
    elif args.test:
        engine.init_engine(is_training=False)
        engine.test()
