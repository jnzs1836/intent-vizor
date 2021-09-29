import argparse
from configs import get_query_focus_config as get_config
from runners import QueryFocusRunner
from data import get_ute_query_loader



if __name__ == '__main__':
    config = get_config(mode='train')
    for i, split in enumerate(config.splits):
        if i not in config.split_ids:
            print("skip {}".format(i))
            continue
        train_keys = split['train_keys']
        valid_keys = split['valid_keys']
        test_keys = split['test_keys']
        train_loaders = [get_ute_query_loader([train_keys[0]], config, shuffle=True, drop_last=True), get_ute_query_loader([train_keys[1]], config, shuffle=True, drop_last=True)]
        valid_loader = get_ute_query_loader([valid_keys[0]], config)
        test_loader = get_ute_query_loader([test_keys[0]], config)
        # train_loader = get_feature_loader(config.video_path, config.splits[i]['train_keys'], config.with_images,
        #                                   config.image_dir, config.video_dir,
        #                                   mapping_file_path=config.mapping_file)
        # test_loader = get_feature_loader(config.video_path, config.splits[i]['test_keys'], config.with_images,
        #                                  config.image_dir, config.video_dir,
        #                                  mapping_file_path=config.mapping_file)
        runner = QueryFocusRunner(config, train_loaders, valid_loader, test_loader, split_id=i)
        runner.build()
        runner.train()
