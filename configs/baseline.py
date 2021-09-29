# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint
import json
from datetime import datetime
import socket
import os


project_dir = Path(__file__).resolve().parent
# dataset_dir = Path('/your/home/directory/data/SumMe/').resolve()
dataset_dir = Path('./out_v2.h5').resolve()
video_list = ["demo"]

save_dir = Path('/your/home/directory/experiments/avs/saves')
score_dir = Path('/your/home/directory/experiments/avs/scores')


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def str2int_list(v):
    str_list = v.split(",")
    int_list = list(map(lambda x: int(x), str_list))
    return int_list


class TestConfig(object):
    def __init__(self, args):
        self.dataset_type = "summe"
        for k, v in args.items():
            setattr(self, k, v)
        self.video_path = Path(self.dataset_dir).joinpath("{}.h5".format(self.dataset_name))
        if self.split_path != "":
            self.split_path = Path(self.split_path)
        else:
            self.split_path = Path(self.dataset_dir).joinpath("{}_splits.json".format(self.dataset_name))
        with open(self.split_path) as fp:
            self.splits = json.load(fp)


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.save_dir = Path(self.save_dir)
        # self.log_dir = self.save_dir
        self.ckpt_path = self.save_dir.joinpath(f'epoch-{self.epoch}.pkl')
        # self.video_path = Path(self.dataset_dir).joinpath("eccv16_dataset_{}_google_pool5.h5".format(self.dataset_name))
        self.video_path = Path(self.dataset_dir).joinpath("{}.h5".format(self.dataset_name))
        if self.split_path != "":
            self.split_path = Path(self.split_path)
        else:
            self.split_path = Path(self.dataset_dir).joinpath("{}_splits.json".format(self.dataset_name))
        with open(self.split_path) as fp:
            self.splits = json.load(fp)
            # self.train_split = splits[0]["train_keys"]
            # self.test_split = splits[0]['test_keys']
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        hostname = socket.gethostname()
        self.run_name = "_".join([self.model_name, self.dataset_name, dt_string, hostname])
        self.run_save_dir = Path(self.save_dir).joinpath(self.run_name)

        self.run_score_dir = Path(self.score_dir).joinpath(self.run_name)
        self.run_log_dir = Path(self.log_dir).joinpath(self.run_name)

        if kwargs["mode"] == "train":
            os.mkdir(self.run_save_dir)
            os.mkdir(self.run_score_dir)
            os.mkdir(self.run_log_dir)
        split_ids = self.split_ids
        if split_ids == "all":
            self.split_ids = [0, 1, 2, 3, 4]
        elif split_ids == "allquery":
            self.split_ids = [0, 1, 2, 3, 4, 5]
        else:
            split_ids = split_ids.split("+")
            split_ids = list(map(lambda x: int(x), split_ids))
            self.split_ids = split_ids
        # self.ckpt_path = self.save_dir.joinpath(f'epoch-{self.epoch}.pkl')


    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=bool, default='true')
    parser.add_argument('--preprocessed', type=bool, default='True')
    parser.add_argument('--video_type', type=str, default='360airballoon')
    parser.add_argument("--model_name", type=str, default="adversarial")
    parser.add_argument("--debug", type=bool, default=False)

    # Data
    parser.add_argument("--dataset_dir", type=str, default="/your/home/directory/data/vsumm/datasets")
    parser.add_argument("--dataset_name", type=str, default="eccv16_dataset_summe_google_pool5")
    parser.add_argument("--dataset_type", type=str, default="summe")
    parser.add_argument("--split_path", type=str, default="")
    parser.add_argument("--with_images", type=bool, default=False)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mapping_file", type=str, default=None)
    # parser.add_argument("--dataset_name", type=str, default="eccv16_dataset_summe_google_pool5.h5")
    # parser.add_argument()
    # Path
    parser.add_argument("--log_dir", type=str, default="/your/home/directory/experiments/video-summarization/runs")
    parser.add_argument("--save_dir", type=str, default="/your/home/directory/experiments/video-summarization/saves")
    # parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--score_dir", type=str, default="/your/home/directory/experiments/video-summarization/scores")

    # Model
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--summary_rate', type=float, default=0.3)

    parser.add_argument("--noise_dim", type=int, default=0)
    parser.add_argument("--best_k", type=int, default=1)
    # Model Type
    parser.add_argument("--summarizer", type=str, default="SUM-GAN")
    parser.add_argument("--discriminator", type=str, default="SUM-GAN")
    parser.add_argument("--compressor", type=str, default="SUM-GAN")
    parser.add_argument("--critic", type=str, default="SUM-GAN")
    parser.add_argument("--solver", type=str, default="GAN")
    parser.add_argument("--compressing_features", type=bool, default=False)
    parser.add_argument("--evaluation_methods", nargs="+", default=["ground_truth_avg", "ground_truth_max"])
    parser.add_argument("--optimal_criteria", type=str, default="f_measure_by_avg")
    parser.add_argument("--cs_m", type=int, default=4)
    parser.add_argument("--stgcn_shortcut", type=int, default=1)
    # Loss
    parser.add_argument("--variance_loss", type=str, default="median") # median ssum, starget, normal
    parser.add_argument("--sparsity_loss", type=str, default="dpp") # dpp, slen

    # Train
    parser.add_argument('--n_epochs', type=int, default=120)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-5)
    parser.add_argument('--discriminator_slow_start', type=int, default=15)
    parser.add_argument('--gt_evaluate', type=bool, default=0)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--discriminator_weight_decay", type=float, default=1)
    parser.add_argument("--discriminator_scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--discriminator_scheduler_gamma", default=0.1)
    # load epoch
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument("--split_ids", type=str, default="all")

    # Knowledge Distillation
    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument("--temperature", type=str, default=10)
    kwargs = parser.parse_args()

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()
