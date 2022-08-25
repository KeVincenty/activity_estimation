import os
import os.path
import numpy as np
import random
import torch
import json
import csv
import pickle
import torch.utils.data as data
from PIL import Image


class CharadesFeatures(data.Dataset):
    def __init__(self,
                 root='/root/workspace/activity_estimation/',
                 mode='train',
                 clip_len=5,
                 ol=None,
                 feature_dir="/root/workspace/SlowFast/vectors/SLOWFAST_8x8_R50_Charades/",
                 label_dir="/root/data/Charades/",
                 class_dir="/root/data/Charades/Charades_v1_classes.txt",
                 ):
        assert mode in ["train", "val", "test"], "Do not support {} mode!".format(mode)
        print("-"*80)
        print("initializing {} set!".format(mode))
        print("-"*80)
        self.root = root
        self.mode = mode
        self.clip_len = clip_len
        self.overlap = ol
        self.feature_dir = feature_dir
        self.label_dir = label_dir + "Charades_v1_test.csv" if mode == "test" else label_dir + "Charades_v1_train.csv"
        self.class_dir = class_dir
        self.fps = 24

        self.classes = {}
        with open(self.class_dir, 'r') as f:
            for line in f:
                self.classes[line[0:4]] = line[4:].strip()

        if self.mode in ["train", "val"]:
            # feature path
            self.feature_dir = os.path.join(self.feature_dir, "train")
            # label info
            self.labels = self.generate_labels_dict("/root/data/Charades/Charades_v1_train.csv") 
            self.data_split = list(self.labels.keys())
            if self.mode == "train":
                self.data_split = self.data_split[:len(self.data_split)*9//10]
            else:
                self.data_split = self.data_split[len(self.data_split)*9//10:]

        else: # test
            self.feature_dir = os.path.join(self.feature_dir, "test")
            self.labels = self.generate_labels_dict("/root/data/Charades/Charades_v1_test.csv") 
            self.data_split = self.data_split = list(self.labels.keys())

        self.num_videos = len(self.data_split)

    def generate_labels_dict(self, file_path):
        labels = {}
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0] == "id":
                    continue
                video_id = line[0]
                feature_path = os.path.join(self.feature_dir, video_id + ".pkl")
                if len(line[9]) == 0 or not os.path.exists(feature_path):
                    continue
                labels[video_id] = [None for _ in range(4)] # [script, actions_info, length, feature_path]
                labels[video_id][0] = line[6]
                labels[video_id][1] = {}
                for action in line[9].split(";"):
                    action_id, start_time, end_time = action.split(" ")
                    labels[video_id][1][action_id] = (int(np.ceil(float(start_time)*self.fps)), int(np.floor(float(end_time)*self.fps)))
                labels[video_id][2] = float(line[-1])
                labels[video_id][3] = feature_path
        return labels

    def clip_sampler(self, vlen):
        num_clips = int(min(np.ceil(vlen / 5), 10))
        if num_clips <= 1:
            return [0]
        elif num_clips < 6:
            ds = 9 // (num_clips - 1)
            seq_idx = [x*ds for x in range(1, num_clips-1)]
            return [0] + seq_idx + [9]
        elif num_clips == 6:
            return [0, 2, 4, 6, 8, 9]
        elif num_clips == 7:
            return [0, 1, 3, 5, 6, 8, 9]
        elif num_clips == 8:
            return [0, 1, 2, 4, 6, 7, 8, 9]
        elif num_clips >= 9:
            return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def load_pickle(self, vpath):
        data_list = []
        with open(vpath, "rb") as f:
            while True:
                try:
                    data_list.append(pickle.load(f))
                except Exception:
                    break
        return data_list

    def __getitem__(self, index):
        video_id = self.data_split[index][:5]
        script, actions, vlen, vpath = self.labels[video_id]
        data_list = self.load_pickle(vpath)
        assert len(data_list) == 30 # the feature is 3(views) * 10(clips)
        clip_idx = self.clip_sampler(vlen)
        actions_list = []
        features_list = []
        for clip in clip_idx:
            features_list.append(torch.cat([x[1] for x in data_list[clip*3:(clip+1)*3]]))
            cur_actions = []
            start_clip = int(data_list[clip*3][0][0][6:12])
            end_clip = int(data_list[clip*3][0][-1][6:12])
            for k,v in actions.items():
                if max(start_clip, v[0]) <= min(end_clip, v[1]):
                    cur_actions.append(int(k[1:]))
            actions_list.append(cur_actions)
        features = torch.stack(features_list, 0)
        return features, actions_list, script # features of each clip, actions within each clip, script for each clip

    def __len__(self):
        # return the number of videos in this dataset
        return self.num_videos


if __name__ == '__main__':
    dataset = CharadesFeatures(mode="train")
    train_loader = torch.utils.data.DataLoader(dataset)
    for features, actions_list, script in train_loader:
        breakpoint()