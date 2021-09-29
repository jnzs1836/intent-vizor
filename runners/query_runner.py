# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import json
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import socket
from utils.summary import generate_summary, evaluate_summary
from pathlib import Path
from utils.metric import Metric
from loggers import Logger, DebugLogger
from factory import build_solver
import gc


class JointIterator():
    def __init__(self, iterators):
        self.iterators = list(map(lambda x: iter(x), iterators))
        self.lens = list(map(lambda x: len(x), self.iterators))
        self.acc_lens = []
        acc = 0
        for l in self.lens:
            acc += l
            self.acc_lens.append(acc)
        self.it = 0
        self.index = 0

    def __update_index__(self):
        if self.index == self.lens[self.it]:
            self.index = 0
            self.it += 1
        if self.it == len(self.iterators):
            raise StopIteration
        
    def __iter__(self):
        return self       
    
    def __next__(self):
        self.__update_index__()
        r = next(self.iterators[self.it])
        self.index += 1
        return r 

    def __len__(self):
        print(len(sum(self.lens)))
        return sum(self.lens)


class QueryFocusRunner(object):
    def __init__(self, config=None, train_loaders=None, valid_loader=None, test_loader=None, split_id=0):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loaders = train_loaders
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.split_id = split_id
        # self.resnet = ResNetFeature().cuda()
        self.solver = build_solver(config)

    def build(self):
        self.run_name = self.config.run_name + "_" + "split-" + str(self.split_id)
        self.save_dir = self.config.run_save_dir.joinpath("split-" + str(self.split_id))
        os.mkdir(self.save_dir)
        self.score_dir = self.config.run_score_dir.joinpath("split-" + str(self.split_id))
        os.mkdir(self.score_dir)
        if self.config.debug:
            self.logger = DebugLogger(os.path.join(self.config.run_log_dir, "split-" + str(self.split_id)), self.save_dir,
                             self.config, self.config.optimal_criteria)
        else:
            self.logger = Logger(os.path.join(self.config.run_log_dir, "split-" + str(self.split_id)), self.save_dir,
                             self.config, self.config.optimal_criteria)

    def train(self):
        step = 0
        valid_step = 0
        test_step = 0

        for epoch_i in trange(self.config.pretrain_embedding_epochs, desc='Pretraining', ncols=80):
            for train_idx, train_loader in enumerate(self.train_loaders):

                for batch_i, batch in enumerate(tqdm(
                        train_loader, desc='train-{}'.format(train_idx), ncols=80, leave=False)):

                    torch.cuda.empty_cache()
                    scores, losses, probs, metrics = self.solver.pretrain_step(batch_i, batch)
                    self.logger.log_pretrain_step(step, scores, losses, probs, metrics)
                    step += 1
            self.logger.log_pretrain_epoch(epoch_i)

        step = 0
        valid_step = 0
        test_step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            gc.collect()
            torch.cuda.empty_cache()
            for train_idx, train_loader in enumerate(self.train_loaders):

                for batch_i, batch in enumerate(tqdm(
                        train_loader, desc='train-{}'.format(train_idx), ncols=80, leave=False)):
                    # for i in range(len(batch)):
                    #     if torch.is_tensor(batch[i]):
                    #         batch[i] = batch[i].cuda()
                    # batch[0] = batch[0].cuda()
                
                    torch.cuda.empty_cache()
                    scores, losses, probs, metrics = self.solver.train_step(batch_i, batch)
                    self.logger.log_train_step(step, scores, losses, probs, metrics)
                    step += 1
            self.logger.log_train_epoch(epoch_i)

            for batch_i, batch in enumerate(tqdm(self.valid_loader, desc="validation")):
                scores, losses, probs, metrics = self.solver.valid_step(batch_i, batch)
                self.logger.log_valid_step(valid_step, scores, losses, probs, metrics)
                valid_step += 1
            self.logger.log_valid_epoch(epoch_i, self.solver.get_model())

            for batch_i, batch in enumerate(tqdm(self.test_loader, desc="test")):
                scores, losses, probs, metrics = self.solver.valid_step(batch_i, batch)
                self.logger.log_test_step(test_step, scores, losses, probs, metrics)
                test_step += 1
            self.logger.log_test_epoch(epoch_i, self.solver.get_model())
            self.solver.scheduler_step(epoch_i)

    def pretrain(self):
        pass


if __name__ == '__main__':
    pass
