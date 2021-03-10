#!/usr/bin/python3.6

import torch
from torch.utils.tensorboard import SummaryWriter
from model import Trainer
from batch_gen import BatchGenerator
import argparse
import random
import time
import os
from eval import evaluate
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# comment out seed to train the model
seed = 1538574472
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', help='two options: train or predict')
parser.add_argument('--dataset', default="breakfast", help='three dataset: breakfast, 50salads, gtea')
parser.add_argument('--split', default='1')

args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 8
lr = 0.0005
num_epochs = 50

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

# Use time data to distinguish output folders in different training
# time_data = '2020-10-15_08-52-26' # turn on this line in evaluation
time_data = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
bz_stages = '/margin_map_both' + time_data
model_dir = "./models/"+args.dataset + bz_stages + "_split_"+args.split
results_dir = "./results/"+args.dataset + bz_stages + "_split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print("{} dataset {} in split {} for single stamp supervision".format(args.action, args.dataset, args.split))
print('batch size is {}, number of stages is {}, sample rate is {}\n'.format(bz, num_stages, sample_rate))

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
writer = SummaryWriter()
trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    # Train the model
    trainer.train(model_dir, batch_gen, writer, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

# Predict the output label for each frame in evaluation and output them
trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
# Read output files and measure metrics (F1@10, 25, 50, Edit, Acc)
evaluate(args.dataset, args.split, time_data)
