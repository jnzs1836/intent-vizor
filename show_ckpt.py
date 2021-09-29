import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='/scratch/gw2145/experiments/query-focused-vsum/saves/TwoStreamGraph_J1KM3_V3.0_WD05_D0_K5_DTL3H10_FD512_TN30_TG2_LR1e4_UTEAllQuery_2021-08-13-02-20-04_gr053.nyu.cluster/split-3/epoch-0.pkl')
args = parser.parse_args()



def load_ckpt(ckpt_path):
    return torch.load(ckpt_path, map_location=torch.device('cpu'))


def print_header(text):
    print()
    print()
    print("----------------------------------------")
    print(text)
    print("----------------------------------------")


def get_attr(model_args, attr):
    if attr in model_args:
        return model_args[attr]
    else:
        return "Unavailable"

def print_args(ckpt):
    model_args = ckpt['args']
    print_header("Training Hyper-Parameters")
    print("learning_rate:              ", model_args['lr'])
    print("weight_decay:               ", model_args['weight_decay'])
    print("scheduler_gamma:            ", model_args['scheduler_gamma'])
    print("scheduler_milestones:       ", "{}".format(model_args['scheduler_milestones']))
    print("warmup_epoch:               ", get_attr(model_args, "warmup_epoch"))
    print("non_linearity_delay:        ", model_args["non_linearity_delay"])

    print_header("Loss Hyper-Parameters")
    print("topic_sparsity_loss_weight: ", model_args['variety_loss_weight'])
    print("topic_variety_loss_weight:  ", model_args['topic_variety_loss_weight'])
    print("prior_loss_weight:          ", model_args['prior_loss_weight'])

    print_header("Topic-Net Hyper-Parameters")
    print("topic_num:                  ", model_args['topic_num'])
    print("topic_embedding_type:       ", model_args['topic_embedding_type'])
    print("topic_embedding_truncation: ", get_attr(model_args, 'topic_embedding_truncation'))
    print("topic_embedding_dim:        ", model_args['topic_embedding_dim'])
    print("topic_net:                  ", model_args['topic_net'])
    print("topic_net_gcn_num_layer:    ", get_attr(model_args, 'topic_net_gcn_num_layer'))
    print("topic_net_query_head:       ", model_args['topic_net_query_attention_head'])
    print("topic_net_hidden_dim:       ", model_args['topic_net_hidden_dim'])

    print_header("Score-Net Hyper-Parameters")
    print("fusion:                     ", model_args['fusion'])
    print("graph_k:                    ", model_args['graph_k'])
    print("local_graph_k:              ", model_args['local_graph_k'])
    print("score_net_gcn_num_layer:    ", get_attr(model_args, 'score_net_gcn_num_layer'))
    print("score_net_mlp_hidden_dim:   ", model_args['score_net_mlp_hidden_dim'])
    print("score_net_mlp_activation:   ", model_args['mlp_activation'])
    print("score_net_similarity:       ", get_attr(model_args, 'score_net_similarity_module'))
    print("local_gcn_mode:             ", get_attr(model_args, 'local_gcn_mode'))
    print("gcn_mode:                   ", get_attr(model_args, 'gcn_mode'))
    print("branch_type:                ", get_attr(model_args, 'branch_type'))

    print_header("Feature Hyper-Parameters")
    print("fast_feature_dim:           ", model_args['fast_feature_dim'])
    print("slow_feature_dim:           ", model_args['slow_feature_dim'])
    print("fusion_feature_dim:         ", model_args['fusion_dim'])


if __name__ == '__main__':
    ckpt = load_ckpt(args.ckpt_path)
    print_args(ckpt)
