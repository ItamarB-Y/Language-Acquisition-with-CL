import torch
import clip
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
import clip
import os
from PIL import Image

clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

prob_index = {"noodles": 0, "napkin": 1, "onion": 2, "knife": 3, "spoon": 4, "pot": 5}
classes = ["noodles", "napkin", "onion", "knife", "spoon", "pot"]

def load_dataset_actual(frames_dir, frames_js_path, noun):
    print("loading dataset...")
    frames_dict = None
    with open(frames_js_path) as js_f:
        frames_dict = json.load(js_f)
    preprc_images_with = []
    preprc_images_without = []
    for filename in os.listdir(frames_dir):
        frame_num = filename.split(".")[0]
        nouns = frames_dict[frame_num]
        image = preprocess(Image.open(frames_dir + filename)).unsqueeze(0).to("cuda")
        if noun in nouns:
            preprc_images_with.append(image)
        else:
            preprc_images_without.append(image)
    random.seed(42)
    dataset_features = preprc_images_with + random.sample(preprc_images_without, len(preprc_images_with))
    dataset_labels = [1 for i in range(len(preprc_images_with))] + [0 for i in range(len(preprc_images_with))]
    X_train, X_test, y_train, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.25, random_state=42, stratify=dataset_labels)
    return X_train, y_train, X_test, y_test

def zero_shot(X_test, y_test, noun):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(X_test)):
        with_token = "a photo with " + noun
        without_token = "a photo with no " + noun
        text = clip.tokenize([with_token, without_token]).to("cuda")
        with torch.no_grad():
            image_features = clip_model.encode_image(X_test[i])
            text_features = clip_model.encode_text(text)
            logits_per_image, logits_per_text = clip_model(X_test[i], text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            if probs[0][0] > probs[0][1]:
                if y_test[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if y_test[i] == 1:
                    fn += 1
                else:
                    tn += 1
    print(noun)
    print(f"tp: {tp}\nfp: {fp}\ntn: {tn}\nfn: {fn}\n")
    print(f"accuracy: {100* ((tp+tn) / (tp+fp+tn+fn))}")


def main():
    for noun in classes:
        train_images, train_labels, test_images, test_labels = load_dataset_actual("/nfs/turbo/coe-chaijy/itamarby/ALA/processed_frames/", "/nfs/turbo/coe-chaijy/itamarby/ALA/frames.json", noun)
        zero_shot(test_images, test_labels, noun)


if __name__ == "__main__":
    main()


"""
loading dataset...
noodles
tp: 63
fp: 54
tn: 240
fn: 230

accuracy: 51.618398637137986
loading dataset...
napkin
tp: 177
fp: 95
tn: 130
fn: 47

accuracy: 68.37416481069042
loading dataset...
onion
tp: 25
fp: 29
tn: 151
fn: 155

accuracy: 48.888888888888886
loading dataset...
knife
tp: 127
fp: 75
tn: 161
fn: 108

accuracy: 61.146496815286625
loading dataset...
spoon
tp: 21
fp: 31
tn: 62
fn: 71

accuracy: 44.86486486486487
loading dataset...
pot
tp: 32
fp: 53
tn: 84
fn: 105

accuracy: 42.33576642335766"""