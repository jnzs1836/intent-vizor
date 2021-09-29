import gc

import torch
import torch.nn as nn
from models.intent_vizor import TopicAwareModel, TopicAbsentModel, TopicAwareShotModel, ShotQueryBaseline, ShotRandomGuess, ShotLinearBaseline
from exceptions import InvalidModelException
from configs import TestConfig, Config



def build_summarizer(config, supervised=False):
    if config.summarizer.startswith ("SUM-GAN"):
        return build_sum_gan_summarizer(config)
    else:
        raise InvalidModelException(config.summarizer)


def build_discriminator(config):
    if config.discriminator == "SUM-GAN":
        return build_sum_gan_discriminator(config)
    else:
        raise InvalidModelException(config.discriminator)



def build_compressor(config):
    if config.compressor == "SUM-GAN":
        return build_sum_gan_compressor(config)
    else:
        raise InvalidModelException(config.compressor)


def build_topic_aware(config):
    return TopicAwareModel(
        device=config.device,
        dropout=config.dropout,
        k=config.graph_k,
        gcn_groups=config.gcn_groups,
        gcn_conv_groups=config.gcn_conv_groups,
        topic_num=config.topic_num,
        topic_embedding_dim=config.topic_embedding_dim,
        fast_feature_dim=config.fast_feature_dim,
        slow_feature_dim=config.slow_feature_dim,
        fusion_dim=config.fusion_dim,
        concept_dim=config.concept_dim,
        topic_net_hidden_dim=config.topic_net_hidden_dim,
        topic_net_num_hidden_layer=config.topic_net_num_hidden_layer,
        topic_net_feature_transformer_head=config.topic_net_feature_transformer_head,
        topic_net_feature_transformer_layer=config.topic_net_feature_transformer_layer,
        topic_net_query_attention_head=config.topic_net_query_attention_head,
        score_net_similarity_dim=config.score_net_similarity_dim,
        score_net_similarity_module_shortcut=config.score_net_similarity_module_shortcut,
        score_net_mlp_num_hidden_layer=config.score_net_mlp_num_hidden_layer,
        score_net_mlp_hidden_dim=config.score_net_mlp_hidden_dim,
        score_net_mlp_init_var=config.score_net_mlp_init_var,
        score_net_norm_layer=config.score_net_norm_layer,
        threshold=config.threshold,
        fusion=config.fusion,
        topic_net=config.topic_net,
        mlp_activation=config.mlp_activation,
        topic_embedding_type=config.topic_embedding_type,
        local_k=config.local_graph_k,
        topic_embedding_truncation=config.topic_embedding_truncation,
        score_net_gcn_num_layer=config.score_net_gcn_num_layer,
        topic_net_gcn_num_layer=config.topic_net_gcn_num_layer,
        score_net_similarity_module=config.score_net_similarity_module,
        branch_type=config.branch_type,
        local_gcn_num_layer=config.local_gcn_num_layer,
        query_decoder_hidden_dim=config.query_decoder_hidden_dim,
        topic_embedding_non_linear_mlp=config.topic_embedding_non_linear_mlp,
        gcn_mode=config.gcn_mode,
        local_gcn_mode=config.local_gcn_mode,
        local_gcn_use_pooling=config.local_gcn_use_pooling
    )


def build_shot_query_topic_aware(config):
    model = TopicAwareShotModel(
        device=config.device,
        dropout=config.dropout,
        k=config.graph_k,
        gcn_groups=config.gcn_groups,
        gcn_conv_groups=config.gcn_conv_groups,
        topic_num=config.topic_num,
        topic_embedding_dim=config.topic_embedding_dim,
        fast_feature_dim=config.fast_feature_dim,
        slow_feature_dim=config.slow_feature_dim,
        fusion_dim=config.fusion_dim,
        concept_dim=config.concept_dim,
        topic_net_hidden_dim=config.topic_net_hidden_dim,
        topic_net_num_hidden_layer=config.topic_net_num_hidden_layer,
        topic_net_feature_transformer_head=config.topic_net_feature_transformer_head,
        topic_net_feature_transformer_layer=config.topic_net_feature_transformer_layer,
        topic_net_query_attention_head=config.topic_net_query_attention_head,
        score_net_similarity_dim=config.score_net_similarity_dim,
        score_net_similarity_module_shortcut=config.score_net_similarity_module_shortcut,
        score_net_mlp_num_hidden_layer=config.score_net_mlp_num_hidden_layer,
        score_net_mlp_hidden_dim=config.score_net_mlp_hidden_dim,
        score_net_mlp_init_var=config.score_net_mlp_init_var,
        score_net_norm_layer=config.score_net_norm_layer,
        threshold=config.threshold,
        fusion=config.fusion,
        topic_net=config.topic_net,
        mlp_activation=config.mlp_activation,
        topic_embedding_type=config.topic_embedding_type,
        local_k=config.local_graph_k,
        topic_embedding_truncation=config.topic_embedding_truncation,
        score_net_gcn_num_layer=config.score_net_gcn_num_layer,
        topic_net_gcn_num_layer=config.topic_net_gcn_num_layer,
        topic_net_shot_query_dim=config.topic_net_shot_query_dim,
        score_net_similarity_module=config.score_net_similarity_module,
        branch_type=config.branch_type,
        local_gcn_num_layer=config.local_gcn_num_layer,
        query_decoder_hidden_dim=config.query_decoder_hidden_dim,
        topic_embedding_non_linear_mlp=config.topic_embedding_non_linear_mlp,
        gcn_mode=config.gcn_mode,
        local_gcn_mode=config.local_gcn_mode,
        local_gcn_use_pooling=config.local_gcn_use_pooling,
        topic_net_attention_num_layer=config.topic_net_attention_num_layer,
        topic_net_attention_mlp_hidden_dim=config.topic_net_attention_mlp_hidden_dim,
    )
    if config.pretrained_score_net:
        pretrained_checkpoint = torch.load(config.pretrained_checkpoint)
        pretrained_config = TestConfig(pretrained_checkpoint['args'])
        pretrained_model = build_topic_aware(pretrained_config)
        pretrained_model.load_state_dict(pretrained_checkpoint['best_state_dict'])
        model.load_pretrained_model(pretrained_model.feature_encoder, pretrained_model.score_net,
                                    pretrained_model.topic_embedding)
        del pretrained_model
        gc.collect()
        torch.cuda.empty_cache()
    return model


def build_shot_query_baseline(config):
    return ShotQueryBaseline(

    )

def build_shot_linear_baseline(config):
    return ShotLinearBaseline(

    )


def build_shot_random_guess(config):
    return ShotRandomGuess()

def build_topic_absent(config):
    return TopicAbsentModel(
        device=config.device,
        dropout=config.dropout,
        k=config.graph_k,
        gcn_groups=config.gcn_groups,
        gcn_conv_groups=config.gcn_conv_groups,
        topic_num=config.topic_num,
        topic_embedding_dim=config.topic_embedding_dim,
        fast_feature_dim=config.fast_feature_dim,
        slow_feature_dim=config.slow_feature_dim,
        fusion_dim=config.fusion_dim,
        concept_dim=config.concept_dim,
        topic_net_hidden_dim=config.topic_net_hidden_dim,
        topic_net_num_hidden_layer=config.topic_net_num_hidden_layer,
        topic_net_feature_transformer_head=config.topic_net_feature_transformer_head,
        topic_net_feature_transformer_layer=config.topic_net_feature_transformer_layer,
        topic_net_query_attention_head=config.topic_net_query_attention_head,
        score_net_similarity_dim=config.score_net_similarity_dim,
        score_net_similarity_module_shortcut=config.score_net_similarity_module_shortcut,
        score_net_mlp_num_hidden_layer=config.score_net_mlp_num_hidden_layer,
        score_net_mlp_hidden_dim=config.score_net_mlp_hidden_dim,
        score_net_mlp_init_var=config.score_net_mlp_init_var
    )
