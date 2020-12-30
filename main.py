import os
import torch
import numpy as np
import pandas as pd
import albumentations
import config
from model import SEResnext50_32x4d
from sklearn import metrics
from dataset import ClassificationDataset
from engine import Engine
from early_stopping import EarlyStopping
import config

import ssl; ssl._create_default_https_context = ssl._create_stdlib_context


def train(fold):
    training_data_path = config.DATA_DIR + "train"
    df = pd.read_csv(config.CSV_PATH + "train.csv")
    device = config.DEVICE
    epochs = config.EPOCHS
    train_bs = config.TRAIN_BATCH_SIZE
    valid_bs = config.EVAL_BATCH_SIZE

    # Train images -> images except the one in current fold
    # Test images -> images in current fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = SEResnext50_32x4d(pretrained="imagenet")
    model.to(device)

    # Known std and mean for image norm
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Define Image Augmentations
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
            # albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            # albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".png") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".png") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=config.NUM_WORKERS
    )

    valid_dataset = ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=config.NUM_WORKERS
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    # Initialize the Engine
    engine = Engine(model=model, optimizer=optimizer, device=device, scheduler=scheduler)
    es = EarlyStopping(patience=5, mode="max")

    for epoch in range(epochs):
        train_loss = engine.train(train_loader)
        predictions, valid_loss = engine.evaluate(valid_loader)
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)

        es(auc, model, model_path=f"model_fold_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break


def predict(fold):
    test_data_path = config.DATA_DIR + "test"
    df = pd.read_csv(config.CSV_PATH + "test.csv")
    device = "cuda"
    model_path=f"model_fold_{fold}.bin"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i + ".png") for i in images]
    targets = np.zeros(len(images))

    test_dataset = ClassificationDataset(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=config.NUM_WORKERS
    )

    model = SEResnext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Okay to pass None to optimizer for prediction step
    engine = Engine(model=model, optimizer=None, device=device)

    predictions = engine.predict(test_loader)
    predictions = np.vstack(predictions).ravel()

    return predictions


if __name__ == '__main__':
    # num_folds = 10
    # for fold in range(num_folds):
    #     train(fold)
    train(0)



