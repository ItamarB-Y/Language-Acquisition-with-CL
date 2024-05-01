import json
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import cv2
import clip
import os
import spacy
from PIL import Image
from ego_utils import obj1_mask, visualize_twohands_obj1

# Change cpature_path, take_path, take_name to run on different takes
dataset_path = '/nfs/turbo/coe-chaijy-unreplicated/datasets/egoexo4d'
capture_path = '/captures/fair_cooking_06/'
take_path = "/takes/fair_cooking_06_2/frame_aligned_videos/"
video_file = 'aria01_214-1.mp4'
take_name = 'fair_cooking_06_2'

# Load spacy and clip
nlp = spacy.load("en_core_web_sm")
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

take_uid = None
vid_length = None

# Find take_uid and take duration in takes.json based on take_name
with open(dataset_path + '/takes.json') as f:
    takes_list = json.load(f)
    for take in takes_list:
        if take['take_name'] == take_name:
            take_uid = take['take_uid']
            vid_length = take['duration_sec']
            break

train_annotations = None
val_annotations = None

# Read train file
with open(dataset_path + '/annotations/atomic_descriptions_train.json') as f:
    train_annotations = json.load(f)['annotations']

# Read validation file
with open(dataset_path + '/annotations/atomic_descriptions_val.json') as f:
    val_annotations = json.load(f)['annotations']

# List of descriptions with timestamp: (timestamp, annotation text)
all_descs = []
if take_uid in train_annotations:
    for anno in train_annotations[take_uid]:
        for desc in anno['descriptions']:
            all_descs.append((desc['timestamp'], desc['text']))
elif take_uid in val_annotations:
    for anno in val_annotations[take_uid]:
        for desc in anno['descriptions']:
            all_descs.append((desc['timestamp'], desc['text']))
else:
    print(f"{take_name} doesn't have annotations")
    sys.exit(1)

# Sort descriptions by timestamp
all_descs.sort(key=lambda x: x[0])

# Returns closest description to a given time
def get_closest_desc(time):
    descs = all_descs
    closest_time_diff = abs(time - descs[0][0])
    closest_desc = descs[0][1]
    for desc in descs:
        curr_time_diff = abs(time - desc[0])
        if curr_time_diff < closest_time_diff:
            closest_time_diff = curr_time_diff
            closest_desc = desc[1]
    if closest_time_diff > 0.5:
        closest_desc = "No Descriptipon"
    return closest_desc


# Nouns to ignore when extracting the nouns from a frame's description
filter_nouns = ['C', 'O', 'hand', 'hands', 'Hand', 'Hands', 'Descriptipon']

# nouns associated with the classes in our dataset
desired_nouns = ["noodles", "knife", "spoon", "napkin", "pot", "onion", "onions"]

cap = cv2.VideoCapture(dataset_path + take_path + video_file)

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("number of frames:", num_frames)

fps = num_frames / vid_length

curr_frame = 0

frames_dict = {}

print("building dataset from frames...")
while(True):
    ret, frame = cap.read()
    if (ret):
        curr_time = curr_frame / fps
        if curr_frame >= 0 and curr_frame <= 20:
            # Grab nouns from closest description to current frame
            desc_display = get_closest_desc(curr_time)
            doc = nlp(desc_display)
            nouns = [token.text for token in doc if 
                     (token.pos_ == 'NOUN' or token.pos_ == 'PROPN') 
                     and token.text not in filter_nouns]
            only_desired_nouns = [noun for noun in nouns if noun in desired_nouns]
            for i in range(len(only_desired_nouns)):
                if only_desired_nouns[i] == "onions":
                    only_desired_nouns[i] = "onion"
            if len(only_desired_nouns) > 0:
                # Save frame temporarily and get egoHOS segmentations 
                frame_path = f"/nfs/turbo/coe-chaijy/itamarby/ALA/processed_frames/{curr_frame}.jpg"
                img_path = f"/home/itamarby/Desktop/EgoExo4D/EgoHOS/mmsegmentation/tmp_imgs/{curr_frame}.jpg"
                cv2.imwrite(img_path, frame)
                seg_obj = obj1_mask(img_path)
                is_saved, vis_path = visualize_twohands_obj1(img_path, seg_obj)
                if (is_saved == "saved"):
                    vis = cv2.imread(vis_path)
                    cv2.imwrite(frame_path, vis)
                    os.remove(vis_path)
                    frames_dict[curr_frame] = only_desired_nouns
                    with open("/nfs/turbo/coe-chaijy/itamarby/ALA/frames_backup.txt", "a") as outf:
                        outf.write(f"{curr_frame},{only_desired_nouns}\n")
                os.remove(img_path)
        else:
            break
        if curr_frame % 500 == 0:
            print(curr_frame)
        curr_frame += 1
    else:
        break
with open("/nfs/turbo/coe-chaijy/itamarby/ALA/frames.json", "w") as outfile:
    json.dump(frames_dict, outfile)
print("frames looked at:", curr_frame)
