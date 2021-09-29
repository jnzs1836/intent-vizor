import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.semantic_evaluation import calculate_semantic_matching, load_videos_tag
from .base_solver import SolverBase
from warmup_scheduler import GradualWarmupScheduler
from .qfvsum_solver import QueryFocusedSolver
# import matlab.engine

# from model import CHAN
# from dataset import UCTDataset
# from utils import load_pickle


class ShotQueryFocusedSolver(QueryFocusedSolver):
    def __init__(self,config=None, score_type="mono", milestones=[1, 10, 30]):
        QueryFocusedSolver.__init__(self, config, score_type, milestones)


    def mono_bce_loss(self, score1, gt_summary):
        assert self.score_type == "mono"
        gt_summary_vec = torch.zeros_like(score1)
        for i in range(score1.size(-1)):
            if i + 1 in gt_summary:
                gt_summary_vec[0, i] = 1

        loss = self.criterion(score1, gt_summary_vec)
        return loss

    def step(self, batch_i, batch, step_type="train"):
        features, shot_query, query_ids, video_ids, concept1_key, concept_2_key, gt_summaries = batch
        features = features.to(device=self.device)
        concept1 = shot_query.to(device=self.device)
        shot_query = shot_query.to(device=self.device)
        
        batch_size = features.shape[0]
        query_id_list = []
        for i in range(batch_size):
            query_id_list.append([])
        for query_id_tensor in query_ids:
            for i in range(query_id_tensor.size(0)):
                query_id_list[i].append(query_id_tensor[i].item())
        # concept2_GT_tmp = concept2_GT[0].masked_select(mask_GT[0]).unsqueeze(0)
        # self.bce_loss(0, 0, concept2_GT_tmp, concept2_GT_tmp, gt_summaries[0])
        self.optimizer.zero_grad()


        score, result_dict = self.model(features, shot_query)
        loss = torch.zeros(1).to(device=self.device)
        precisions = []
        recalls = []
        f1s = []
        scores = []

        var_loss = self.prob_variety_loss(result_dict['topic_probs'])
        topic_var_loss = self.topic_variety_loss(result_dict['all_scores'])
        prob_variance_loss = self.prob_variance_loss(result_dict['topic_probs'])
        for i in range(batch_size):
            video_idx = int(video_ids[i][1:]) - 1
            pred_score = score[i][: len(self.videos_tag[video_idx])]
            pred_score_tmp = pred_score.unsqueeze(0)
            loss += self.mono_bce_loss(pred_score_tmp, gt_summaries[i])
            masked_score = pred_score.detach().cpu()
            masked_score[query_id_list[i]] = 0
            _, top_index = pred_score.topk(int(pred_score.shape[0] * self.top_percent))
            gt_summary = gt_summaries[i]
            gt_summary = list(map(lambda x: x-1, gt_summary))
            masked_gt_summary = list(filter(lambda x: x not in query_id_list[i], gt_summary))
            precision, recall, f1 = calculate_semantic_matching(top_index.cpu(), masked_gt_summary, self.videos_tag[video_idx])
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            scores.append(pred_score)

        loss += var_loss.sum() + topic_var_loss.sum() + result_dict['prior_loss'] * self.prior_loss_weight\
                + prob_variance_loss.sum()
        if step_type == "train":
            loss.backward()
            self.optimizer.step()
        metrics = {
            "f1": sum(f1s) / batch_size,
            "precision": sum(precisions) / batch_size,
            "recall": sum(recalls) / batch_size
        }
        scores = {

        }
        losses = {
            "loss": loss,
            "topic_prob_variance": prob_variance_loss.sum(),
            "topic_prob_variety": var_loss.sum(),
            "topic_score_variance": topic_var_loss.sum(),
            "prior_loss": result_dict['prior_loss'] * self.prior_loss_weight,

        }
        probs = {

        }
        return scores, losses, probs, metrics

    def train_step(self, batch_i, batch):
        self.model.train()
        return self.step(batch_i, batch)

    def valid_step(self, batch_i, batch):
        self.model.eval()
        with torch.no_grad():
            return self.step(batch_i, batch, "valid")

    def get_model(self):
        return self.model
