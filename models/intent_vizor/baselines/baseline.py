import torch
import torch.nn as nn
from models.intent_vizor.score_net import ScoreNet
from models.intent_vizor.intent_net.topic_net import TopicNet
from models.intent_vizor.feature_encoder.conv_feature_encoder import FeatureEncoder
class TopicAwareModel(nn.Module):
    def __init__(self, device="cuda", topic_num=20, topic_embedding_dim=64, score_net_hidden_dim=64, concept_dim=300, topic_net_hidden_dim=128,
                 max_segment_num=20, max_frame_num=200,
                 ):
        nn.Module.__init__(self)
        self.device = device
        self.feature_encoder = FeatureEncoder(device=device, topic_num=topic_num, topic_embedding_dim=topic_embedding_dim, hidden_dim=score_net_hidden_dim)
        self.score_net = ScoreNet(device=device, topic_num=topic_num, topic_embedding_dim=topic_embedding_dim, hidden_dim=score_net_hidden_dim)
        self.topic_net = TopicNet(device=device, topic_num=topic_num, concept_dim=concept_dim, hidden_dim=topic_net_hidden_dim)
        self.topic_num = topic_num
        self.max_segment_num = max_segment_num
        self.max_frame_num = max_frame_num
        self.non_linear = nn.ReLU()

    
    def get_frame_features(self, batch, seg_len):
        mask=torch.zeros(batch.size(0), batch.size(1),batch.size(2),dtype=torch.bool).to(device=self.device)
        for i in range(seg_len.size(0)):
            for j in range(len(seg_len[i])):
                for k in range(seg_len[i][j]):
                    mask[i][j][k]=1
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, batch.size(3))
        frame_features = batch.masked_select(mask)
        frame_features = frame_features.view( batch.size(0), -1, batch.size(3))

        return frame_features 

    def forward(self, batch, seg_len, concept1, concept2):
        batch_size = batch.size(0)
        overall_score = torch.zeros(batch_size, self.max_segment_num, self.max_frame_num).to(device=self.device)
        frame_features = self.get_frame_features(batch, seg_len)
        video_features = self.feature_encoder(batch, frame_features, seg_len, concept1, concept2)
        topic_probs = self.topic_net(batch, seg_len, concept1, concept2, video_features)

        for topic_id in range(self.topic_num):
            topic_prob = topic_probs[:, topic_id]
            topic = torch.ones(batch_size, dtype=torch.long) * topic_id
            topic = topic.to(device=self.device)
            topic_score, _ = self.score_net(batch, seg_len, concept1, concept2, topic, video_features)
            # topic_score = topic_score * topic_prob
            topic_score = torch.mul(topic_score, topic_prob.view(batch_size, 1, 1))

            topic_score -= 0.01
            # print(topic_score.size())
            overall_score += self.non_linear(topic_score)
        overall_score /= self.topic_num
        return overall_score, overall_score
