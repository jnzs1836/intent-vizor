# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import math


# dynamic graph from knn
def knn(x, y=None, k=10):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    if y is None:
        y = x
    # logging.info('Size in KNN: {} - {}'.format(x.size(), y.size()))
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return idx


def get_temporal_graph(x, seg_lens, max_seg_length=200):
    batch_size = x.size(0)
    max_seg_num = seg_lens.size(1)
    attention_mask = torch.zeros(batch_size, max_seg_num, int(max_seg_length / 4), dtype=torch.bool).to(
        device=x.device)
    neighbor_idx = []

    for i in range(batch_size):
        neighbor_idx.append([])
        last_valid = 0
        skip_seg = 0
        for j in range(len(seg_lens[i])):
            for k in range(max_seg_length // 4):
                neighbor_idx[i].append([])
                base_idx = i * max_seg_num * max_seg_length // 4
                idx = j * max_seg_length // 4 + k
                # print(len(neighbor_idx[i]))
                # print(j * max_seg_length//4 + k, j, k)
                if k < math.ceil(seg_lens[i][j] / 4.0):
                    neighbor_idx[i][j * max_seg_length // 4 + k].append(base_idx + last_valid)
                    neighbor_idx[i][j * max_seg_length // 4 + k].append(base_idx + idx)
                    neighbor_idx[i][j * max_seg_length // 4 + k].append(base_idx + idx)
                    last_valid = idx
                    # if i == 1 and idx == 81:
                        # print("test", j, k, math.ceil(seg_lens[i][j] / 4.0))
                    if k == 0 and j != 0:
                        neighbor_idx[i][skip_seg][2] = base_idx + idx
                    if k == math.ceil(seg_lens[i][j] / 4.0) - 1:
                        skip_seg = idx
                    else:
                        neighbor_idx[i][j * max_seg_length // 4 + k][2] = base_idx + idx + 1

                else:
                    # neighbor_idx[i][skip_seg][2] = base_idx + idx
                    neighbor_idx[i][j * max_seg_length // 4 + k].append(base_idx + idx)
                    neighbor_idx[i][j * max_seg_length // 4 + k].append(base_idx + idx)
                    neighbor_idx[i][j * max_seg_length // 4 + k].append(base_idx + idx)
    # print(len(neighbor_idx))
    # for i in range(batch_size):
    #     print(len(neighbor_idx[i]), len(seg_lens[i]))
    neighbor_idx = torch.LongTensor(neighbor_idx).to(device=x.device)
    # print("neighbor: ", neighbor_idx.size())
    x_ = x.transpose(1, 2)
    # print("x_", x_.size())
    # print(batch_size * max_seg_num * max_seg_length // 4)
    temporal_graph = x_.contiguous().view(batch_size * max_seg_num * max_seg_length // 4, -1)[neighbor_idx, :]
    temporal_graph = temporal_graph.permute(0, 3, 1, 2)
    # temporal_graph = x[neighbor_idx, :]
    # print("tg:", temporal_graph.size())

    # print(seg_lens[1] // 4)
    # print("test:", neighbor_idx[1][81])
    # print("test tg:", temporal_graph[1, 3, 81, :])
    # print("test x:", x[1, 3, 100])
    # 1 / 0
    # print("seg:", seg_lens)
    return temporal_graph

# get graph feature
def get_graph_feature(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    # print("graph x:", x.size())
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    # print(idx_knn.size())
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    # print("x:", x.size())
    # print("idx:", idx.size())
    # print("feature:", feature.size())
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn


# basic block
class GCNeXt(nn.Module):
    def __init__(self, channel_in, channel_out, k=10, norm_layer=None, groups=32, width_group=4, idx=None):
        super(GCNeXt, self).__init__()
        self.k = k
        self.groups = groups

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups
        # self.tconvs = nn.Sequential(
        #     nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
        #     nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
        #     nn.Conv1d(width, channel_out, kernel_size=1),
        # ) # temporal graph
        self.tconvs = nn.Sequential(
            nn.Conv2d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=(1, 3), groups=groups, padding=0), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        )  # temporal graph
        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        ) # semantic graph

        self.relu = nn.ReLU(True)
        self.idx_list = idx

    def forward(self, x, seg_lens):
        # print(x.size())
        # print(seg_lens.size())
        # print(x.size())
        identity = x  # residual
        x_t = get_temporal_graph(x, seg_lens)
        tout = self.tconvs(x_t)  # conv on temporal graph
        tout = tout.max(dim=-1, keepdim=False)[0]
        # print("tout:", tout.size())
        x_f, idx = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        # print("xf: ",x_f.size())
        # print("idx: ", idx.size())
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)

        # print("sout:", sout.size())
        out = tout + identity + sout  # fusion
        if not self.idx_list is None:
            self.idx_list.append(idx)
        return self.relu(out)

class EgoGCNeXt(nn.Module):
    def __init__(self, channel_in, channel_out, topic_channel,k=10, norm_layer=None, groups=32, width_group=4, idx=None):
        super(EgoGCNeXt, self).__init__()
        self.k = k
        self.groups = groups
        self.topic_channel = topic_channel
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups

        self.tconvs = nn.Sequential(
            nn.Conv2d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=(1, 3), groups=groups, padding=0), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        )  # temporal graph
        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2 + topic_channel, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        )  # semantic graph

        self.relu = nn.ReLU(True)
        self.idx_list = idx

    def forward(self, x, seg_lens, topic_features):
        identity = x  # residual
        x_t = get_temporal_graph(x, seg_lens)
        tout = self.tconvs(x_t)  # conv on temporal graph
        tout = tout.max(dim=-1, keepdim=False)[0]
        x_f, idx = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        xf_pad = torch.zeros(topic_features.size(0), topic_features.size(1), x_f.size(2), x_f.size(3)).to(device=x.device)
        topic_pad = torch.zeros(topic_features.size(0), x_f.size(1), 1, 1).to(device=x.device)
        topic_features = topic_features.unsqueeze(-1)
        topic_features_padded = torch.cat([topic_pad, topic_features], dim=1)
        topic_features_padded = topic_features_padded.expand(-1, -1, x_f.size(2), 1)


        x_f_padded = torch.cat([x_f, xf_pad], dim=1)
        merged_graph = torch.cat([x_f_padded, topic_features_padded], dim=3)
        sout = self.sconvs(merged_graph)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)

        out = tout + identity + sout  # fusion
        if not self.idx_list is None:
            self.idx_list.append(idx)
        return self.relu(out)


class GTAD(nn.Module):
    def __init__(self, opt):
        super(GTAD, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.feat_dim = opt["feat_dim"]
        self.bs = opt["batch_size"]
        self.h_dim_1d = 256
        self.h_dim_2d = 128
        self.h_dim_3d = 512
        self.goi_style = opt['goi_style']
        self.h_dim_goi = self.h_dim_1d*(16,32,32)[opt['goi_style']]
        self.idx_list = []

        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list),
        )

        # Regularization
        self.regu_s = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32),
            nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        )
        self.regu_e = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32),
            nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        )

        # Backbone Part 2
        self.backbone2 = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32,idx=self.idx_list),
        )

        # SGAlign: sub-graph of interest alignment

        # Localization Module
        self.localization = nn.Sequential(
            nn.Conv2d(self.h_dim_goi, self.h_dim_3d, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_3d, self.h_dim_2d, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, self.h_dim_2d, kernel_size=opt['kern_2d'], padding=opt['pad_2d']), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, self.h_dim_2d, kernel_size=opt['kern_2d'], padding=opt['pad_2d']), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, 2, kernel_size=1), nn.Sigmoid()
        )

        # Position encoding (not used)
        self.pos = torch.arange(0, 1, 1.0 / self.tscale).view(1, 1, self.tscale)

    def forward(self, snip_feature):
        del self.idx_list[:]  # clean the idx list
        base_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        gcnext_feature = self.backbone2(base_feature)  #

        regu_s = self.regu_s(base_feature).squeeze(1)  # start
        regu_e = self.regu_e(base_feature).squeeze(1)  # end

        if self.goi_style==2:
            idx_list = [idx for idx in self.idx_list if idx.device == snip_feature.device]
            idx_list = torch.cat(idx_list, dim=2)
        else:
            idx_list = None



if __name__ == '__main__':
    import opts
    from torchsummary import summary
    opt = opts.parse_opt()
    opt.k = 10
    opt = vars(opt)
    model = GTAD(opt).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    input = torch.randn(4, 2048, 100).cuda()
    a, b, c = model(input)
    # print(a.shape, b.shape, c.shape)

    summary(model, (400,100))
    '''
    Total params: 9,495,428
    Trainable params: 9,495,428
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.15
    Forward/backward pass size (MB): 1398.48
    Params size (MB): 36.22
    Estimated Total Size (MB): 1434.85
    '''
