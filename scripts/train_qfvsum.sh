cd ../
python train_qfvsum.py \
    --split_path=/your/home/directory/data/UTE/all_splits.json \
    --feature_dir=/your/home/directory/data/UTE/UTC_feature/data/processed \
    --save_dir=/your/home/directory/experiments/query-focused-vsum/saves \
    --log_dir=/your/home/directory/experiments/query-focused-vsum/runs \
    --score_dir=/your/home/directory/experiments/query-focused-vsum/scores \
    --dictionary_path=/your/home/directory/data/UTE/UTC_feature/data/processed/query_dictionary.pkl \
    --device=cuda --annotation_dir=/your/home/directory/data/UTE/UTC_feature/data/origin_data \
    --dataset_name UTEAllQuery --model_name=model \
    --batch_size 2 --weight_decay 0.6 --dropout 0 --lr 1e-4 --graph_k 8 --score_net_similarity_module_shortcut False \
    --fusion_dim 512 --split_ids 3 --scheduler_milestone 10,30,75 --topic_net_query_attention_head 5 \
    --topic_net_feature_transformer_layer 3 --topic_net_num_hidden_layer 4 --topic_net_hidden_dim 2048 \
    --score_net_mlp_hidden_dim 1024 --score_net_mlp_num_hidden_layer 2 --topic_embedding_dim 128 --topic_num 20 \
    --score_net_mlp_init_var 1 --variety_loss_weight 0 --slow_feature_dim 1024 --fast_feature_dim 256 --gcn_conv_groups 4\
     --score_net_norm_layer batch --threshold 0.02 --fusion graph --topic_net graph --mlp_activation relu \
     --topic_variety_loss_weight 0 --non_linearity_delay 20 --fusion local_gcn --topic_embedding_type vanilla \
     --prior_loss_weight 0 --prob_variance_loss_weight 0 --warmup_epoch 8 --topic_embedding_truncation 2 \
     --score_net_gcn_num_layer 3 --topic_net_gcn_num_layer 3 --local_graph_k 10 \
     --score_net_similarity_module inner_product --branch_type dual --pretrain_embedding_epochs 0 \
     --topic_embedding_non_linear_mlp True --n_epochs 121 --shot_score_variety_loss_weight 0 --gcn_mode mutual_map \
     --local_gcn_mode mutual_map --local_gcn_use_pooling False
