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



class UTEQueryDataset(Dataset):
    def __init__(self, h5path="", split=None, transform=None, with_images=False, image_dir=None, video_dir=None,
                 mapping_file_path=None, oracle_summaries=[], feature_dir = "./data/processed",
                 dictionary_path="./data/processed/query_dictionary.pkl",
                 shot_tag_dir = "./data/origin_data/Dense_per_shot_tags", gt_summaries={}
                 ):
        self.feature_dir = feature_dir
        self.gt_summaries = gt_summaries
        self.split = split
        self.dataset = oracle_summaries
        self.video_dir = video_dir
        self.dictionary_path = dictionary_path
        # for video_id in self.split:
        #     for _ , _, files in os.walk("./data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
        #         for file in files:
        #             self.dataset.append(file[:file.find("_oracle.txt")]+"_"+str(video_id))
        self.embedding = {}
        if self.dictionary_path.endswith("txt"):
            with open(self.dictionary_path) as fp:
                for line in fp.readlines():
                    splits = line.strip().split()
                    if len(splits) != 301:
                        continue
                    word = splits[0]
                    vector = splits[1:]
                    vector = list(map(lambda x: float(x), vector))
                    self.embedding[word] = vector
        else:
            self.embedding=load_pickle(self.dictionary_path)
        self.shot_tag_dir = shot_tag_dir
        self.max_segment_num = 20
        self.max_frame_num = 200
    def __getitem__(self,index):
        video_id=self.dataset[index].split('_')[2]
        f=h5py.File(os.path.join(self.feature_dir, 'V'+video_id[2:]+'_resnet_avg.h5'),'r')
        features=f['features'][()]
        seg_len=f['seg_len'][()]

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
        concept1_embedding=torch.Tensor(self.embedding[concept1])
        concept2_embedding=torch.Tensor(self.embedding[concept2])
        gt_summary = self.gt_summaries[record_entry]
        return features,seg_len,concept1_embedding,concept2_embedding,concept1_GT,concept2_GT,mask_GT, \
               video_id, concept1, concept2, gt_summary

    def __len__(self):
        return len(self.dataset)


def select_default(item):
    return item[0:10]


def collate_fn(batch):
    tmp = list(map(lambda x: select_default(x), batch))
    tmp = default_collate(tmp)
    gt_summaires = []
    for item in batch:
        gt_summaires.append(item[10])
    # print(batch[0])
    # print(batch)
    return *tmp, gt_summaires

def get_ute_query_loader(videos, config, drop_last=False, shuffle=False):
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
    data_loader = DataLoader(UTEQueryDataset(oracle_summaries=oracle_summaries, dictionary_path=config.dictionary_path,
                                             shot_tag_dir=shot_tag_dir, feature_dir=config.feature_dir, gt_summaries=gt_summaries
                                             ), shuffle=shuffle,collate_fn=collate_fn, batch_size=config.batch_size, drop_last=drop_last)
    return data_loader


def get_video_query_loader(video_feature_dir, config, drop_last=False, shuffle=False):
    oracle_summaries = [
        "Street_Men_P01", "Men_Men_P01", "Car_Car_P01", "Car_Street_P01", "Car_Signal_P01", "Signal_Light_P01",
        "Sign_Road_P01", "Sign_Board_P01"
    ]
    shot_tag_dir = os.path.join(config.annotation_dir, "Dense_per_shot_tags")
    gt_summaries = {}
    for entry in oracle_summaries:
        gt_summaries[entry] = [1, 2, 3]
    print(video_feature_dir)
    data_loader = DataLoader(UTEQueryDataset(oracle_summaries=oracle_summaries, dictionary_path=config.dictionary_path,
                                             shot_tag_dir=shot_tag_dir, feature_dir=video_feature_dir,
                                             gt_summaries=gt_summaries
                                             ), shuffle=shuffle, collate_fn=collate_fn, batch_size=config.batch_size,
                             drop_last=drop_last)
    return data_loader
