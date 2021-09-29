import os
import h5py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.semantic_evaluation import calculate_semantic_matching, load_videos_tag
from .base_solver import SolverBase
from .qfvsum_solver import QueryFocusedSolver
# import matlab.engine

# from model import CHAN
# from dataset import UCTDataset
# from utils import load_pickle


class QueryFocusedDeploymentSolver(QueryFocusedSolver):
    def __init__(self,config=None, score_type="mono", milestones=[1, 10, 30]):
        self.config=config
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.config["gpu"]
        # self._build_dataloader()
        # self._bulid_model()
        # self._build_optimizer()
        self.max_f1=0
        self.max_p=0
        self.max_r=0
        self.top_percent = 0.02
        self.device = self.config.device
        self.videos_tag = load_videos_tag()
        self.score_type = score_type
        self.max_segment_num = 20
        self.max_frame_num = 200
        self.milestones = milestones
    def build(self, model):
        self.criterion = torch.nn.BCELoss()
        self.model = model.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                          milestones=self.milestones,
                                                        gamma=self.config.scheduler_gamma)
        # current_work_dir = os.getcwd()
        # os.chdir("./evaluation_code")
        # self.evaluator = matlab.engine.start_matlab()
        # os.chdir(current_work_dir)
        pass


    def test_step(self, batch_i, batch):
        features, seg_len, concept1, concept2, concept1_GT, concept2_GT, mask_GT, video_ids, concept1_keys, concept2_keys, gt_summaries = batch
        features = features.to(device=self.device)
        concept1 = concept1.to(device=self.device)
        concept2 = concept2.to(device=self.device)
        scores, result_dict = self.model(features, seg_len.to(device=self.device), concept1, concept2)
        result = []
        batch_size = features.size(0)
        for i in range(batch_size):
            score = scores[i]
            _, top_index = score.topk(int(score.shape[0] * self.top_percent))
            # print(result_dict['all_scores'].size())
            print(result_dict['all_scores'][0][0])
            print(result_dict['all_scores'][0][1])
            print(result_dict['all_scores'].squeeze(0).size())
            data_item = {
                "summary": top_index.cpu().tolist(),
                "video": video_ids[i],
                "scores": score.detach().cpu().tolist(),
                "topic_probs": result_dict['topic_probs'][i].detach().cpu().tolist(),
                "topic_scores": result_dict['all_scores'].squeeze(0).detach().cpu().tolist(),
                "concept1": concept1_keys[i],
                "concept2": concept2_keys[i]
            }
            result.append(data_item)
        return result


    def get_model(self):
        return self.model

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()
