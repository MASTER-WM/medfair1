import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from netcal.metrics import ECE
import wandb
from MEDFAIR.parse_args import collect_args

from models.CFair import CFair
from models.DomainInd import DomainInd
from models.EnD import EnD
from models.GroupDRO import GroupDRO
from models.LAFTR import LAFTR
from models.LNL import LNL
from models.ODR import ODR
from models.resampling import resampling
from models.SWAD import SWAD

# وارد کردن API key برای لاگین به wandb
wandb.login(key="7c3adfae8d2f0585fa2a18defe84b89702d0a5ba")


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['Path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['binaryLabel']
        protected_attr = self.dataframe.iloc[idx]['Age_Category']

        if self.transform:
            image = self.transform(image)

        protected_attr = torch.tensor(protected_attr)

        return image, label, protected_attr


def calculate_fairness_metrics(y_true, y_pred, protected_attr):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    FPR = fp / (fp + tn)
    FNR = fn / (fn + tp)
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    ece = ECE().measure(y_pred, y_true)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'BCE': (FPR + FNR) / 2,
        'ECE': ece,
        'TPR@80': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'EqOdd': (FPR + FNR) / 2
    }

    return metrics


def preprocess_data(metadata_path, image_dir):
    metadata = pd.read_excel(metadata_path)
    metadata.dropna(inplace=True)

    train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

    def save_images_as_pickle(data, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for idx, row in data.iterrows():
            image_path = os.path.join(image_dir, row['Image_filename'])
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
            with Image.open(image_path) as img:
                img = img.resize((224, 224))
                img_array = np.array(img)
                pickle_file = os.path.join(save_dir, f"{row['Image_filename']}.pkl")
                with open(pickle_file, 'wb') as f:
                    pickle.dump(img_array, f)

    save_images_as_pickle(train_data, 'train_images/')
    save_images_as_pickle(val_data, 'val_images/')
    save_images_as_pickle(test_data, 'test_images/')

    return train_data, val_data, test_data


def main():
    opt, wandb = collect_args()

    # مسیر به متادیتا و دایرکتوری تصاویر
    metadata_path = 'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx'
    image_dir = '/Users/amir/PycharmProjects/Medfair/MEDFAIR/BrEaST-Lesions_USG-images_and_masks'

    # پیش‌پردازش داده‌ها و ذخیره تصاویر به صورت فایل‌های پیککل
    train_data, val_data, test_data = preprocess_data(metadata_path, image_dir)

    # تنظیم مسیر فایل‌های پیککل به عنوان image_feature_path
    data_setting = {
        'image_feature_path': {
            'train': 'train_images/',
            'val': 'val_images/',
            'test': 'test_images/'
        },
        # سایر تنظیمات...
    }

    # تعریف transform برای پیش‌پردازش تصاویر
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(dataframe=train_data, transform=transform)
    val_dataset = CustomDataset(dataframe=val_data, transform=transform)
    test_dataset = CustomDataset(dataframe=test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=opt['batch_size'], shuffle=False)

    model_names = ['CFair', 'DomainInd', 'EnD', 'GroupDRO', 'LAFTR', 'LNL', 'ODR', 'Resampling', 'SWAD']
    models_list = [CFair(opt, wandb), DomainInd(opt, wandb), EnD(opt, wandb), GroupDRO(opt, wandb),
                   LAFTR(opt, wandb), LNL(opt, wandb), ODR(opt, wandb), resampling(opt, wandb), SWAD(opt, wandb)]

    final_results = pd.DataFrame()

    for model_name, model in zip(model_names, models_list):
        if opt['resume_path']:
            model.load_state_dict(torch.load(opt['resume_path']))
            print(f"Model {model_name} loaded from {opt['resume_path']}")

        optimizer = optim.SGD(model.parameters(), lr=opt['lr'], momentum=0.9)
        trained_model = train_model(model, {'train': train_loader, 'val': val_loader}, nn.CrossEntropyLoss(), optimizer,
                                    opt, num_epochs=opt['total_epochs'])

        save_path = os.path.join(opt['save_folder'], f"{model_name}_final.pth")
        torch.save(trained_model.state_dict(), save_path)
        print(f"Model {model_name} saved to {save_path}")

        for age_group in train_data['Age_Category'].unique():
            group_data = train_data[train_data['Age_Category'] == age_group]
            y_true = group_data['binaryLabel']
            y_pred = trained_model.predict(group_data['Path'])
            metrics = calculate_fairness_metrics(y_true, y_pred, age_group)
            metrics['Model'] = model_name
            metrics['Group'] = f'Grp. {age_group}'
            metrics_df = pd.DataFrame([metrics])
            final_results = pd.concat([final_results, metrics_df], ignore_index=True)

    final_results = final_results.pivot(index='Model', columns='Group')
    print(final_results)

    final_results.to_csv('final_results.csv')


if __name__ == "__main__":
    main()