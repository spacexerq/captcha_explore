import albumentations
import numpy as np
import pandas as pd
import os
import glob
import random
from pprint import pprint
from tqdm import tqdm
import seaborn as sns

import torch
from torch import nn
from torch.nn import functional as F

from PIL import Image
from PIL import ImageFile

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Configurations for the files
DIR = "../dataset/"
BATCH_SIZE = 16
IMG_HEIGHT = 75
IMG_WIDTH = 300
EPOCHS = 100
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {DEVICE}")
warnings.filterwarnings("ignore", category=UserWarning)


def show_random_images(df, column_name):
    f = plt.figure(figsize=(10, 10))
    for i in range(16):
        i += 1
        ax = f.add_subplot(4, 4, i)
        sample = random.choice(df[column_name])
        image = mpimg.imread(sample)
        ax.set_title(sample.split("/")[-1])
        plt.imshow(image)


def get_loss_function(x, bs, targets):
    log_softmax_values = F.log_softmax(x, 2)

    input_lengths = torch.full(
        size=(bs,), fill_value=log_softmax_values.size(0), dtype=torch.int32
    )

    target_lengths = torch.full(
        size=(bs,), fill_value=targets.size(1), dtype=torch.int32
    )

    return nn.CTCLoss(blank=0)(log_softmax_values, targets, input_lengths, target_lengths)


class MyCaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(MyCaptchaModel, self).__init__()

        # CNN Layer
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # RNN Layer Preprocess
        self.linear1 = nn.Linear(1152, 64)
        self.drop1 = nn.Dropout(0.2)

        # LSTM GRU
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, channel, height, width = images.size()

        x = F.relu(self.conv1(images))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)

        x = self.linear1(x)
        x = self.drop1(x)

        x, _ = self.gru(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)

        if targets is not None:
            loss = get_loss_function(x, bs, targets)
            return x, loss

        return x, None


def train_function(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(DEVICE)

        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

    return fin_loss / len(data_loader)


def eval_function(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(DEVICE)

            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)

        return fin_preds, fin_loss / len(data_loader)


def encode_targets():
    # Load images from files
    image_files = glob.glob(DIR + "*.jpg")
    targets_orig = [x.split("\\")[-1].split(".")[0] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]  # squeeze

    # Encode images
    lbl_enc = LabelEncoder()
    lbl_enc.fit(targets_flat)

    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc) + 1  # transform to np and remove 0 index

    return image_files, targets_enc, targets_orig, lbl_enc


class DatasetClassifier:
    def __init__(self, image_paths, targets, resize=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(always_apply=True)
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        target = self.targets[item]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.long)
        }


def early_stopping(patience, count, prev_loss, current_loss, threshold):
    if abs(prev_loss - current_loss) < threshold and count >= patience:
        return "stop"
    elif abs(prev_loss - current_loss) < threshold:
        return "count"
    else:
        return False


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("-")
            else:
                temp.append(encoder.inverse_transform([k])[0])
        tp = "".join(temp)
        cap_preds.append(tp)
    return cap_preds


if __name__ == '__main__':
    paths = []
    labels = []
    for image in os.listdir(DIR):
        paths.append(os.path.join(DIR, image))
        labels.append(image.split(".")[0])

    df = pd.DataFrame({
        "paths": paths,
        "labels": labels
    })

    df.head()

    # Train-test split
    image_files, targets_enc, targets_orig, lbl_enc = encode_targets()

    (train_imgs, test_imgs, train_targets, test_targets, _, test_orig_targets) = train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=0, shuffle=True)

    ImageFile.LoadTruncatedImages = True

    # Classify images, load using pytorch's DataLoader
    train_dataset = DatasetClassifier(
        image_paths=train_imgs, targets=train_targets, resize=(IMG_HEIGHT, IMG_WIDTH)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    test_dataset = DatasetClassifier(
        image_paths=test_imgs, targets=test_targets, resize=(IMG_HEIGHT, IMG_WIDTH)
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )

    # Load the models
    model = MyCaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(DEVICE)

    # Create optimizer and callbacks
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    patience = 6
    count = 0
    prev_train_loss = 0
    threshold = 0.05
    loss = []

    for epoch in range(EPOCHS):
        train_loss = train_function(model, train_loader, optimizer)
        valid_preds, valid_loss = eval_function(model, test_loader)
        valid_cap_preds = []

        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_cap_preds.extend(current_preds)

        pprint(list(zip(test_orig_targets, valid_cap_preds))[15:20])
        print(f"Epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")

        res = early_stopping(patience, count, prev_train_loss, train_loss, threshold)

        loss.append(train_loss)

        if res == "stop":
            print("Early Stopping Implemented.")
            final_epoch = epoch
            break
        elif res == "count" and train_loss < 0.2:
            count += 1
            print(f"Patience at {patience - count}")
        else:
            prev_train_loss = train_loss
        torch.save(model.state_dict(), "models/model_" + str(epoch) + ".bin")

        df_pytorch = pd.DataFrame({"loss": loss})
        plt.figure(figsize=(15, 5))
        plt.grid()
        plt.xlabel("Epoch No.")
        plt.ylabel("Loss Value")
        plt.title("Loss During Epoch Training")
        sns.lineplot(data=df_pytorch, palette="tab10", linewidth=2.5)
        plt.savefig("final.png")
        plt.close("All")
        # plt.show()
    torch.save(model.state_dict(), "../model.bin")
