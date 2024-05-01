import os
import torch
import clip
import json
import random
import torchvision.transforms as TT
from PIL import Image
from models import CustomClip
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import AttentionDataset

clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

classes = ["noodles", "napkin", "onion", "knife", "spoon", "pot"]

def get_predictions(logits):
    return torch.argmax(logits, dim=1)

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

def tune_clip(train_images, train_labels, test_images, test_labels, noun):
    print("tunning for ", noun)
    train_dataset = AttentionDataset(train_images, train_labels)
    test_dataset = AttentionDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)  

    model = CustomClip(clip_model=clip_model)
    if torch.cuda.is_available():
        model.to("cuda")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

    num_epochs = 2
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.to("cuda")

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f"train loss at epoch {epoch}: {train_loss}")

    with torch.no_grad():
        test_loss = 0
        num = 0
        total = 0
        tp = 0
        for inputs, labels in test_loader:
            num += 1
            labels = labels.to("cuda")
            outputs = model(inputs)
            preds = get_predictions(outputs)
            tp += labels.shape[0] - torch.count_nonzero(preds - labels).item()
            loss = criterion(outputs, labels)
            total += outputs.shape[0]
            test_loss += loss
        print(f"test loss: {test_loss}")
        print(f"accuracy: {(tp / total) * 100}")
    return

def main():
    for noun in classes:
        train_images, train_labels, test_images, test_labels = load_dataset_actual("/nfs/turbo/coe-chaijy/itamarby/ALA/processed_frames/", "/nfs/turbo/coe-chaijy/itamarby/ALA/frames.json", noun)
        tune_clip(train_images, train_labels, test_images, test_labels, noun)

if __name__ == "__main__":
    main()



"""
tunning for  noodles
train loss at epoch 0: 26.87147608399391
train loss at epoch 1: 21.348112404346466
test loss: 8.599186897277832
accuracy: 76.32027257240205
loading dataset...
tunning for  napkin
train loss at epoch 0: 18.709319338202477
train loss at epoch 1: 12.937744140625
test loss: 4.427017688751221
accuracy: 87.52783964365256
loading dataset...
tunning for  onion
train loss at epoch 0: 17.156876623630524
train loss at epoch 1: 12.852234125137329
test loss: 3.918597936630249
accuracy: 88.61111111111111
loading dataset...
tunning for  knife
train loss at epoch 0: 20.263443008065224
train loss at epoch 1: 13.95596032589674
test loss: 4.254449367523193
accuracy: 88.95966029723992
loading dataset...
tunning for  spoon
train loss at epoch 0: 10.839142382144928
train loss at epoch 1: 8.544019937515259
test loss: 2.8712151050567627
accuracy: 75.13513513513513
loading dataset...
tunning for  pot
train loss at epoch 0: 12.679078221321106
train loss at epoch 1: 8.26033154129982
test loss: 3.1328086853027344
accuracy: 83.57664233576642
"""