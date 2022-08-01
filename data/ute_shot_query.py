import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import h5py
import json
import pickle
import torch

def load_json(filename):
    with open(filename, encoding='utf8') as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def save_pickle(object,filename):
    with open(filename, 'wb') as f:
        pickle.dump(object,f)



class UTEShotQueryDataset(Dataset):
    def __init__(self, h5path="", split=None, transform=None, with_images=False, image_dir=None, video_dir=None,
                 mapping_file_path=None, oracle_summaries=[], feature_dir = "./data/processed",
                 dictionary_path="./data/processed/query_dictionary.pkl",
                 shot_tag_dir = "./data/origin_data/Dense_per_shot_tags", gt_summaries={},
                 shot_query_path="./data/ute_query/shot_query.json"
                 ):
        self.feature_dir = feature_dir
        self.gt_summaries = gt_summaries
        self.split = split
        self.dataset = oracle_summaries
        self.video_dir = video_dir
        self.dictionary_path = dictionary_path
        self.shot_query_path = shot_query_path
        with open(shot_query_path) as fp:
            self.shot_query_index = json.load(fp)
        # for video_id in self.split:
        #     for _ , _, files in os.walk("./data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
        #         for file in files:
        #             self.dataset.append(file[:file.find("_oracle.txt")]+"_"+str(video_id))
        self.embedding=load_pickle(self.dictionary_path)
        self.shot_tag_dir = shot_tag_dir
        self.max_segment_num = 20
        self.max_frame_num = 200

    def get_frame_features(self, batch, seg_len):
        mask=torch.zeros(batch.size(0), batch.size(1),dtype=torch.bool)
        for i in range(seg_len.size(0)):
            for j in range(seg_len[i]):
                mask[i][j]=1
        mask = mask.unsqueeze(-1).expand(-1, -1, batch.size(2))
        frame_features = batch.masked_select(mask)
        frame_features = frame_features.view(-1, batch.size(2))
        return frame_features

    def __getitem__(self,index):
        video_id=self.dataset[index].split('_')[2]
        f=h5py.File(os.path.join(self.feature_dir, 'V'+video_id[2:]+'_resnet_avg.h5'),'r')
        features=f['features'][()]

        seg_len=f['seg_len'][()]
        features = torch.FloatTensor(features)
        seg_len = torch.LongTensor(seg_len)
        frame_features = self.get_frame_features(features, seg_len)
        transfer={"Cupglass":"Glass",
                  "Musicalinstrument":"Instrument",
                  "Petsanimal":"Animal"}
        record_entry = self.dataset[index]
        concept1,concept2=self.dataset[index].split('_')[0:2]

        concept1_GT = torch.zeros(self.max_segment_num * self.max_frame_num)
        concept2_GT = torch.zeros(self.max_segment_num * self.max_frame_num)
        with open( os.path.join(self.shot_tag_dir,  "P"+video_id[1:]+"/P"+video_id[1:]+".txt"),"r") as f:
            lines=f.readlines()
            for index,line in enumerate(lines):
                concepts=line.strip().split(',')
                if concept1 in concepts:
                    concept1_GT[index] = 1
                if concept2 in concepts:
                    concept2_GT[index] = 1

        shot_num=seg_len.sum()
        mask_GT=torch.zeros(self.max_segment_num * self.max_frame_num, dtype=torch.bool)
        for i in range(shot_num):
            mask_GT[i]=1

        if concept1 in transfer:
            concept1=transfer[concept1]
        if concept2 in transfer:
            concept2=transfer[concept2]
        concept1_embedding=self.embedding[concept1]
        concept2_embedding=self.embedding[concept2]
        shot_query = self.shot_query_index[record_entry]
        query_features = frame_features[shot_query]
        gt_summary = self.gt_summaries[record_entry]
        return frame_features, query_features, shot_query,video_id, concept1, concept2, gt_summary

    def __len__(self):
        return len(self.dataset)


def select_default(item):
    return item[0:6]


def collate_fn(batch):
    tmp = list(map(lambda x: select_default(x), batch))
    tmp = default_collate(tmp)
    gt_summaires = []
    for item in batch:
        gt_summaires.append(item[6])
    # print(batch[0])
    # print(batch)
    return *tmp, gt_summaires

def get_ute_shot_query_loader(videos, config, drop_last=False, shuffle=False):
    oracle_summaries = []
    gt_summaries = {}
    oracle_summary_dir = os.path.join(config.annotation_dir, "Query-Focused_Summaries/Oracle_Summaries")
    for video_id in videos:
        video_dir = os.path.join(oracle_summary_dir,  str(video_id))
        for _, _, files in os.walk(video_dir):
            for file in files:
                gt_summary = []
                record_entry = file[:file.find("_oracle.txt")] + "_" + str(video_id)
                oracle_summaries.append(record_entry)
                with open(os.path.join(video_dir, file),
                          "r") as f:
                    for line in f.readlines():
                        gt_summary.append(int(line.strip()))
                gt_summaries[record_entry] = gt_summary
    # print(oracle_summaries)
    shot_tag_dir = os.path.join(config.annotation_dir, "Dense_per_shot_tags")
    data_loader = DataLoader(UTEShotQueryDataset(oracle_summaries=oracle_summaries, dictionary_path=config.dictionary_path,
                                             shot_tag_dir=shot_tag_dir, feature_dir=config.feature_dir, gt_summaries=gt_summaries
                                             ), shuffle=shuffle,collate_fn=collate_fn, batch_size=config.batch_size, drop_last=drop_last)
    return data_loader


if __name__ == '__main__':
    class DataConfig:
        def __init__(self):
            self.annotation_dir = "/mnt/d/Data/UTE/UTC_feature/data/origin_data"
            self.dictionary_path = "/mnt/d/Data/UTE/UTC_feature/data/processed/query_dictionary.pkl"
            self.feature_dir = "/mnt/d/Data/UTE/UTC_feature/data/processed"
            self.batch_size = 4
    config = DataConfig()
    for video in ["P01", "P02", "P03", "P04"]:
        print("processing", video)
        train_loaders = get_ute_shot_query_loader([video], config, shuffle=True, drop_last=True)
    for loader in train_loaders:
        for batch in loader:
            print(batch)
