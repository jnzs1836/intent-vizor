import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.semantic_evaluation import calculate_semantic_matching, load_videos_tag
from .base_solver import SolverBase
from warmup_scheduler import GradualWarmupScheduler
import gc
# import matlab.engine

# from model import CHAN
# from dataset import UCTDataset
# from utils import load_pickle


class QueryFocusedSolver(SolverBase):
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
        self.variety_loss_weight = config.variety_loss_weight
        self.topic_variety_loss_weight = config.topic_variety_loss_weight
        self.non_linearity_delay = config.non_linearity_delay
        self.max_segment_num = 20
        self.max_frame_num = 200
        self.milestones = milestones
        self.prior_loss_weight = config.prior_loss_weight
        self.topic_prob_variance_loss_weight = config.prob_variance_loss_weight
        self.pretrain_embedding_epochs = config.pretrain_embedding_epochs
        self.pretrain_embedding_lr = config.pretrain_embedding_lr
        self.pretrain_embedding_weight_decay = config.pretrain_embedding_weight_decay
        self.shot_score_variety_loss_weight = config.shot_score_variety_loss_weight

    def build(self, model):
        self.criterion = torch.nn.BCELoss()
        self.model = model.to(device=self.device)

        parameters = self.model.parameters()
        if self.config.summarizer == "baseline" or self.config.summarizer == "linear_baseline":
            pass
        elif not self.config.fine_tune_score_net and self.config.pretrained_score_net:
            parameters = self.model.topic_net.parameters()

        self.optimizer = torch.optim.Adam(parameters, lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                          milestones=self.milestones,
                                                        gamma=self.config.scheduler_gamma)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=self.config.warmup_multiplier,
                                                total_epoch=self.config.warmup_epoch,
                                                after_scheduler=self.base_scheduler)

        if self.config.summarizer != "baseline" and self.config.summarizer != "linear_baseline":
            self.pretrain_embedding_optimizer = optim.Adam(list(self.model.topic_net.parameters()) +
                                                       list(self.model.query_decoder.parameters()),
                                                       lr=self.pretrain_embedding_lr,
                                                       weight_decay=self.pretrain_embedding_weight_decay
                                                       )

        # current_work_dir = os.getcwd()
        # os.chdir("./evaluation_code")
        # self.evaluator = matlab.engine.start_matlab()
        # os.chdir(current_work_dir)
        pass


    def _bulid_model(self):
        self.model = CHAN(self.config).cuda()

    def _build_dataset(self):
        return UCTDataset(self.config)

    def _build_dataloader(self):
        dataset=self._build_dataset()
        self.dataloader=DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["num_workers"])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def output(self):
        print(" max_p = ",self.max_p," max_r = ",self.max_r," max_f1 = ",self.max_f1)

    def prob_variety_loss(self, topic_probs):
        if topic_probs is None:
            return torch.Tensor([0.0]).to(self.device)
        # sample = torch.ones_like(topic_probs)
        # mean = torch.mean(topic_probs, dim=1)
        # var = torch.var(topic_probs, dim=1) 
        loss = -torch.var(topic_probs, dim=1) * self.variety_loss_weight / (0.0001 + torch.mean(topic_probs, dim=1))
        return loss

    def scheduler_step(self, epoch_i):
        self.scheduler.step(epoch_i)

    def prob_variance_loss(self, topic_probs, epsilon=1e-4):
        if topic_probs is None:
            return torch.Tensor([0.0]).to(self.device)
        # median_tensor = torch.zeros(topic_probs.shape[0]).to(topic_probs.device)
        # median_tensor.fill_(torch.median(topic_probs))
        median_tensor = topic_probs.median(dim=1)[0]
        median_tensor = median_tensor.unsqueeze(1).expand(-1, topic_probs.size(1))
        loss = nn.MSELoss(reduction="none")
        variance = loss(topic_probs, median_tensor)
        variance = torch.mean(variance, dim=1)
        # variance = torch.sum(variance, dim=1) / topic_probs.size(1)
        return self.topic_prob_variance_loss_weight * 1 / (variance + epsilon)

    def prob_sparsity_loss(self, topic_probs):
        if topic_probs is None:
            return 0
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(topic_probs, dim=1) - 0.1)

    def topic_variety_loss(self, all_scores):
        if all_scores is None:
            return torch.Tensor([0.0]).to(self.device)
        loss = -torch.var(all_scores, dim=1).mean(dim=1) * self.topic_variety_loss_weight
        return loss

    def shot_score_variety_loss(self, all_scores):
        if all_scores is None:
            return 0
        loss = -torch.var(all_scores, dim=1).mean() #* self.shot_score_variety_loss_weight
        return loss

    def concept_bce_loss(self, concept1, concept2, decoded_encoding):
        concept = torch.cat([concept1, concept2], dim=1)
        loss = torch.dist(concept, decoded_encoding)
        return loss

    def bce_loss(self, score1, score2, concept1_GT, concept2_GT, gt_summary):
        gt_score = (concept1_GT + concept2_GT) / 2
        # batch_size = score1.size(0)
        if self.score_type == "mono":
            gt_summary_vec = torch.zeros_like(score1)
            # print(gt_summary)
            for i in range(score1.size(-1)):
                if i + 1 in gt_summary:
                    gt_summary_vec[0, i] = 1

            # print("gt vecs:", gt_summary_vecs.sum(dim=1))
            # print(gt_summaries)
            # print("concept avg:", gt_score)
            # print(gt_summary_vec.size())
            # print(concept1_GT.size())
            # print(gt_summary_vec)
            loss = self.criterion(score1, gt_summary_vec)
            return loss
        else:
            loss1 = self.criterion(score1, concept1_GT.to(device=self.device))
            loss2 = self.criterion(score2, concept2_GT.to(device=self.device))
            return loss1 + loss2

    def activate_non_linearity(self):
        self.model.activate_non_linearity()


    def get_result(self, batch_size, f1s, precisions, recalls, loss, prob_variance_loss, var_loss, topic_var_loss, result_dict):
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

    def pretrain_step(self, batch_i, batch):
        return self.step(batch_i, batch, "pretrain")

    def step(self, batch_i, batch, step_type="train"):
        gc.collect()
        torch.cuda.empty_cache()
        features, seg_len, concept1, concept2, concept1_GT, concept2_GT, mask_GT, video_ids, concept1_key, concept_2_key, gt_summaries = batch
        features = features.to(device=self.device)
        concept1 = concept1.to(device=self.device)
        concept2 = concept2.to(device=self.device)
        concept1_GT = concept1_GT.to(device=self.device)
        concept2_GT = concept2_GT.to(device=self.device)
        mask_GT = mask_GT.to(device=self.device)
        train_num = seg_len.shape[0]
        # concept2_GT_tmp = concept2_GT[0].masked_select(mask_GT[0]).unsqueeze(0)
        # self.bce_loss(0, 0, concept2_GT_tmp, concept2_GT_tmp, gt_summaries[0])
        self.optimizer.zero_grad()
        self.pretrain_embedding_optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        batch_size = seg_len.shape[0]
        mask = torch.zeros(train_num, self.config.max_segment_num, self.config.max_frame_num,
                           dtype=torch.bool).to(device=self.device)
        for i in range(train_num):
            for j in range(len(seg_len[i])):
                for k in range(seg_len[i][j]):
                    mask[i][j][k] = 1

        # batch_size * max_seg_num * max_seg_length

        if step_type == "pretrain":
            decoded_query = self.model.forward_decoder(features, seg_len.to(device=self.device), concept1, concept2)
            loss = self.concept_bce_loss(concept1, concept2, decoded_query)
            loss.backward()
            self.pretrain_embedding_optimizer.step()
            return {}, {}, {
                "bce_loss": loss.mean(),
            }, {}

        concept1_score, result_dict = self.model(features, seg_len.to(device=self.device), concept1, concept2)

        loss = torch.zeros(1).to(device=self.device)
        precisions = []
        recalls = []
        f1s = []
        scores = []

        var_loss = self.prob_variety_loss(result_dict['topic_probs'])
        topic_var_loss = self.topic_variety_loss(result_dict['all_scores'])
        shot_score_variety_loss = self.shot_score_variety_loss(result_dict['all_scores'])
        prob_variance_loss = self.prob_variance_loss(result_dict['topic_probs'])
        for i in range(batch_size):
            video_idx = int(video_ids[i][1:]) - 1
            # concept1_score_tmp = concept1_score[i].masked_select(mask[i]).unsqueeze(0)
            # concept2_score_tmp = concept2_score[i].masked_select(mask[i]).unsqueeze(0)
            pred_score = concept1_score[i][: len(self.videos_tag[video_idx])]
            concept1_score_tmp = pred_score.unsqueeze(0)
            # concept2_score_tmp = concept2_score[i].unsqueeze(0)
            concept1_GT_tmp = concept1_GT[i].masked_select(mask_GT[i]).unsqueeze(0)
            concept2_GT_tmp = concept2_GT[i].masked_select(mask_GT[i]).unsqueeze(0)
            # loss1 = self.criterion(concept1_score_tmp, concept1_GT_tmp.to(device=self.device))
            # loss2 = self.criterion(concept2_score_tmp, concept2_GT_tmp.to(device=self.device))
            # loss += loss1 + loss2
            loss += self.bce_loss(concept1_score_tmp, concept1_score_tmp, concept1_GT_tmp, concept2_GT_tmp, gt_summaries[i])
            score = pred_score + pred_score
            # score = score.masked_select(mask[i])
            # score = score
            _, top_index = score.topk(int(score.shape[0] * self.top_percent))
            gt_summary = gt_summaries[i]
            gt_summary = list(map(lambda x: x-1, gt_summary))
            precision, recall, f1 = calculate_semantic_matching(top_index.cpu(), gt_summary, self.videos_tag[video_idx])
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            scores.append(score)

        loss += var_loss.sum() + topic_var_loss.sum() + result_dict['prior_loss'] * self.prior_loss_weight\
                + prob_variance_loss.sum() + shot_score_variety_loss
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
            "shot_score_variety_loss": shot_score_variety_loss,
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



    def train(self):
        print("start to evaluate random result")
        self.evaluate(self.config["test_video"],self.config["top_percent"])
        print("end to evaluate random result")

        criterion=torch.nn.BCELoss()

        for epoch in range(self.config["epoch"]):
            batch_count=0
            self.evaluate(self.config["test_video"],self.config["top_percent"])

    def evaluate(self,video_id,top_percent):
        current_work_dir = os.getcwd()
        os.chdir("./evaluation_code")
        evaluator = matlab.engine.start_matlab()
        os.chdir(current_work_dir)

        f1=0
        p=0
        r=0

        embedding=load_pickle("./data/processed/query_dictionary.pkl")

        evaluation_num=0

        for _,_,files in os.walk("./data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
            evaluation_num=len(files)
            for file in files:
                summaries_GT=[]
                with open("./data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)+"/"+file,"r") as f:
                    for line in f.readlines():
                        summaries_GT.append(int(line.strip()))
                f=h5py.File('./data/processed/V'+str(video_id)+'_resnet_avg.h5','r')
                features=torch.tensor(f['features'][()]).unsqueeze(0).cuda()
                seg_len=torch.tensor(f['seg_len'][()]).unsqueeze(0).cuda()

                transfer={"Cupglass":"Glass","Musicalinstrument":"Instrument","Petsanimal":"Animal"}

                concept1,concept2=file.split('_')[0:2]
                # concept1, concept2 = self.dataset[index].split('_')[0:2]
                if concept1 in transfer:
                    concept1=transfer[concept1]
                if concept2 in transfer:
                    concept2=transfer[concept2]

                concept1=torch.tensor(embedding[concept1]).unsqueeze(0).cuda()
                concept2=torch.tensor(embedding[concept2]).unsqueeze(0).cuda()

                mask=torch.zeros(1,self.config["max_segment_num"],self.config["max_frame_num"],dtype=torch.bool).cuda()
                for i in range(1):
                    for j in range(len(seg_len[i])):
                        for k in range(seg_len[i][j]):
                            mask[i][j][k]=1
                concept1_score,concept2_score=self.model(features,seg_len,concept1,concept2)

                score=concept1_score+concept2_score

                score=score.masked_select(mask)

                _,top_index=score.topk(int(score.shape[0]*top_percent))

                top_index+=1

                out=evaluator.eval111(matlab.int32(list(top_index.cpu().numpy())),matlab.int32(summaries_GT),video_id)

                f1+=out["f1"]
                r+=out["rec"]
                p+=out["pre"]

        if f1/evaluation_num>self.max_f1:
            self.max_f1=f1/evaluation_num
            self.max_p=p/evaluation_num
            self.max_r=r/evaluation_num

        print("p = ",p/evaluation_num," r = ",r/evaluation_num," f1 = ",f1/evaluation_num)

    def get_model(self):
        return self.model
