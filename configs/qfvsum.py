# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pprint
import json
from datetime import datetime
import socket
import os
from .baseline import Config, str2bool, str2int_list

project_dir = Path(__file__).resolve().parent
# dataset_dir = Path('/your/home/directory/data/SumMe/').resolve()
dataset_dir = Path('./out_v2.h5').resolve()
video_list = ["demo"]

save_dir = Path('/your/home/directory/experiments/avs/saves')
score_dir = Path('/your/home/directory/experiments/avs/scores')


def get_query_focus_config(parse=True, **optional_kwargs):
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrained_score_net", type=str2bool, default=False)
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--fine_tune_score_net", type=str2bool, default=True)
    # Data
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset_dir", type=str, default="/your/home/directory/data/vsumm/datasets")
    parser.add_argument("--feature_dir", type=str, default="/your/home/directory/Data/UTE/UTC_feature/data/processed")
    parser.add_argument("--dictionary_path", type=str,
                        default="/your/home/directory/Data/UTE/UTC_feature/data/processed/query_dictionary.pkl")
    parser.add_argument("--annotation_dir", type=str, default="/your/home/directory/Data/UTE/UTC_feature/data/origin_data")
    parser.add_argument("--dataset_name", type=str, default="eccv16_dataset_summe_google_pool5")
    parser.add_argument("--dataset_type", type=str, default="summe")
    parser.add_argument("--split_path", type=str, default="/your/home/directory/Data/UTE/splits.json")
    parser.add_argument("--with_images", type=bool, default=False)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mapping_file", type=str, default=None)
    parser.add_argument("--max_segment_num", type=int, default=20)
    parser.add_argument("--max_frame_num", type=int, default=200)
    # parser.add_argument("--dataset_name", type=str, default="eccv16_dataset_summe_google_pool5.h5")
    # parser.add_argument()
    # Path
    parser.add_argument("--log_dir", type=str, default="/your/home/directory/code/c2smart/experiments/logs/runs")
    parser.add_argument("--save_dir", type=str, default="/your/home/directory/code/c2smart/experiments/logs/saves")
    # parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--score_dir", type=str, default="/your/home/directory/code/c2smart/experiments/logs/scores")

    # Model
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--summary_rate', type=float, default=0.3)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument("--intent_dropout", type=int, default=0)
    parser.add_argument("--concept_dim", type=int, default=300)

    parser.add_argument("--warmup_multiplier", type=float, default=1)
    parser.add_argument("--warmup_epoch", type=float, default=0)

    parser.add_argument("--pretrain_embedding_epochs", type=int, default=0)
    parser.add_argument("--pretrain_embedding_lr", type=float, default=1e-4)
    parser.add_argument("--pretrain_embedding_weight_decay", type=float, default=0.5)
    parser.add_argument("--query_decoder_hidden_dim", type=int, default=256)

    parser.add_argument("--mlp_activation", type=str, default="relu")
    parser.add_argument("--non_linearity_delay", type=int, default=0)
    parser.add_argument("--branch_type", type=str, default="dual")
    parser.add_argument("--feature_encoder", type=str, default="fast_slow")

    parser.add_argument("--gcn_mode", type=str, default="cat")
    parser.add_argument("--local_gcn_mode", type=str, default="cat")
    parser.add_argument("--local_gcn_use_pooling", type=str2bool, default=False)

    parser.add_argument("--topic_num", type=int, default=20)
    parser.add_argument("--topic_embedding_dim", type=int, default=256)
    parser.add_argument("--topic_embedding_type", type=str, default="vanilla")
    parser.add_argument("--topic_embedding_truncation", type=float, default=0)
    parser.add_argument("--topic_embedding_non_linear_mlp", type=str2bool, default=False)

    parser.add_argument("--topic_net", type=str, default="transformer")
    parser.add_argument("--topic_net_hidden_dim", type=int, default=128)
    parser.add_argument("--topic_net_num_hidden_layer", type=int, default=2)
    parser.add_argument("--topic_net_gcn_num_layer", type=int, default=1)

    parser.add_argument("--topic_net_feature_transformer_head", type=int, default=8)
    parser.add_argument("--topic_net_feature_transformer_layer", type=int, default=3)
    parser.add_argument("--topic_net_query_attention_head", type=int, default=4)

    # Shot Query Topic-Net Only
    parser.add_argument("--topic_net_attention_num_layer", type=int, default=4)
    parser.add_argument("--topic_net_attention_mlp_hidden_dim", type=int, default=256)
    # parser.add_argument("--topic_net_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--topic_net_shot_query_dim", type=int, default=256)


    parser.add_argument("--fast_feature_dim", type=int, default=512)
    parser.add_argument("--slow_feature_dim", type=int, default=2048)
    parser.add_argument("--fusion_dim", type=int, default=512)

    parser.add_argument("--score_net_similarity_dim", type=int, default=1024)
    parser.add_argument("--score_net_similarity_module_shortcut", type=str2bool, default=False)
    parser.add_argument("--score_net_mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--score_net_mlp_num_hidden_layer", type=int, default=2)
    parser.add_argument("--score_net_mlp_init_var", type=float, default=0.0025)
    parser.add_argument("--score_net_norm_layer", type=str, default="batch")
    parser.add_argument("--score_net_gcn_num_layer", type=int, default=1)
    parser.add_argument("--score_net_similarity_module", type=str, default="inner_product")

    parser.add_argument("--score_branch_net", type=str, default="gcn")
    parser.add_argument("--topic_branch_net", type=str, default="gcn")

    parser.add_argument("--local_gcn_num_layer", type=int, default=1)

    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--fusion", type=str, default="dconv")

    parser.add_argument('--graph_k', type=int, default=6)
    parser.add_argument('--local_graph_k', type=int, default=6)

    parser.add_argument("--gcn_groups", type=int, default=32)
    parser.add_argument("--gcn_conv_groups", type=int, default=4)


    parser.add_argument("--noise_dim", type=int, default=0)
    parser.add_argument("--best_k", type=int, default=1)
    # Model Type
    parser.add_argument("--summarizer", type=str, default="TopicAware")
    parser.add_argument("--discriminator", type=str, default="SUM-GAN")
    parser.add_argument("--compressor", type=str, default="SUM-GAN")
    parser.add_argument("--critic", type=str, default="SUM-GAN")
    parser.add_argument("--solver", type=str, default="QueryFocus-MonoScore")
    parser.add_argument("--compressing_features", type=bool, default=False)
    parser.add_argument("--evaluation_methods", nargs="+", default=["f1"])
    parser.add_argument("--optimal_criteria", type=str, default="f1")
    parser.add_argument("--cs_m", type=int, default=4)
    parser.add_argument("--stgcn_shortcut", type=int, default=1)
    # Loss
    parser.add_argument("--prior_loss_weight", type=float, default=0.5)  # median ssum, starget, normal
    parser.add_argument("--variety_loss_weight", type=float, default=100) # median ssum, starget, normal
    parser.add_argument("--topic_variety_loss_weight", type=float, default=0)  # median ssum, starget, normal
    parser.add_argument("--prob_variance_loss_weight", type=float, default=0)
    parser.add_argument("--shot_score_variety_loss_weight", type=float, default=0)
    parser.add_argument("--variance_loss", type=str, default="median") # median ssum, starget, normal
    parser.add_argument("--sparsity_loss", type=str, default="dpp") # dpp, slen

    # Train
    parser.add_argument('--n_epochs', type=int, default=120)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--discriminator_lr', type=float, default=1e-5)
    parser.add_argument('--discriminator_slow_start', type=int, default=15)
    parser.add_argument('--gt_evaluate', type=bool, default=0)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--discriminator_weight_decay", type=float, default=1)
    parser.add_argument("--discriminator_scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    parser.add_argument("--scheduler_milestones", type=str2int_list, default="1,10,30")

    parser.add_argument("--discriminator_scheduler_gamma", default=0.1)
    # load epoch
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument("--split_ids", type=str, default="allquery")

    # Knowledge Distillation
    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument("--temperature", type=str, default=10)
    kwargs = parser.parse_args()

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
