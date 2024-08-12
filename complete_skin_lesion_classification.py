
import h5py
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import time

# Additional imports for deep learning models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# read metadata
path = 'your_path/fariness_data/HAM10000/'

demo_data = pd.read_csv(path + 'HAM10000_metadata.csv')
print(Counter(demo_data['dataset']))

# add image path to the metadata
pathlist = demo_data['image_id'].values.tolist()
paths = ['HAM10000_images/' + i + '.jpg' for i in pathlist]
demo_data['Path'] = paths

# remove age/sex == null 
demo_data = demo_data[~demo_data['age'].isnull()]
demo_data = demo_data[~demo_data['sex'].isnull()]

# unify the value of sensitive attributes
sex = demo_data['sex'].values
sex[sex == 'male'] = 'M'
sex[sex == 'female'] = 'F'
demo_data['Sex'] = sex

# split subjects to different age groups
demo_data['Age_multi'] = demo_data['age'].values.astype('int')
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(-1,19), 0, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(20,39), 1, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(40,59), 2, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(60,79), 3, demo_data['Age_multi'])
demo_data['Age_multi'] = np.where(demo_data['Age_multi']>=80, 4, demo_data['Age_multi'])

demo_data['Age_binary'] = demo_data['age'].values.astype('int')
demo_data['Age_binary'] = np.where(demo_data['Age_binary'].between(-1, 60), 0, demo_data['Age_binary'])
demo_data['Age_binary'] = np.where(demo_data['Age_binary']>= 60, 1, demo_data['Age_binary'])

# convert to binary labels
# benign: bcc, bkl, dermatofibroma, nv, vasc
# malignant: akiec, mel

labels = demo_data['dx'].values.copy()
labels[labels == 'akiec'] = '1'
labels[labels == 'mel'] = '1'
labels[labels != '1'] = '0'

labels = labels.astype('int')

demo_data['binaryLabel'] = labels

def split_811(all_meta, patient_ids):
    sub_train, sub_val_test = train_test_split(patient_ids, test_size=0.2, random_state=0)
    sub_val, sub_test = train_test_split(sub_val_test, test_size=0.5, random_state=0)
    train_meta = all_meta[all_meta.lesion_id.isin(sub_train)]
    val_meta = all_meta[all_meta.lesion_id.isin(sub_val)]
    test_meta = all_meta[all_meta.lesion_id.isin(sub_test)]
    return train_meta, val_meta, test_meta

sub_train, sub_val, sub_test = split_811(demo_data, np.unique(demo_data['lesion_id']))

sub_train.to_csv('your_path/fariness_data/HAM10000/split/new_train.csv')
sub_val.to_csv('your_path/fariness_data/HAM10000/split/new_val.csv')
sub_test.to_csv('your_path/fariness_data/HAM10000/split/new_test.csv')

# you can have a look of some examples here
img = cv2.imread('your_path/fariness_data/HAM10000/HAM10000_images/ISIC_0027419.jpg')
print(img.shape)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

test_meta = pd.read_csv('your_path/fariness_data/HAM10000/split/new_train.csv')

path = 'your_path/fariness_data/HAM10000/pkls/'
images = []
start = time.time()
for i in range(len(test_meta)):
    img = cv2.imread(test_meta.iloc[i]['Path'])
    # resize to the input size in advance to save time during training
    img = cv2.resize(img, (256, 256))
    images.append(img)
end = time.time()
print("Time taken to load and resize images: ", end-start)
with open(path + 'train_images.pkl', 'wb') as f:
    pickle.dump(images, f)

# Define a custom dataset class
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

        if self.transform:
            image = self.transform(image)

        return image, label

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = CustomDataset(dataframe=sub_train, transform=transform)
val_dataset = CustomDataset(dataframe=sub_val, transform=transform)
test_dataset = CustomDataset(dataframe=sub_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the models
model_resnet18 = models.resnet18(pretrained=True)
num_ftrs = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(num_ftrs, 2)

model_vgg16 = models.vgg16(pretrained=True)
num_ftrs = model_vgg16.classifier[6].in_features
model_vgg16.classifier[6] = nn.Linear(num_ftrs, 2)

# Function to train the model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model

# Train the models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_resnet18 = model_resnet18.to(device)
model_vgg16 = model_vgg16.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_resnet18 = optim.SGD(model_resnet18.parameters(), lr=0.001, momentum=0.9)
optimizer_vgg16 = optim.SGD(model_vgg16.parameters(), lr=0.001, momentum=0.9)

dataloaders = {
    'train': train_loader,
    'val': val_loader
}

model_resnet18 = train_model(model_resnet18, dataloaders, criterion, optimizer_resnet18, num_epochs=25)
model_vgg16 = train_model(model_vgg16, dataloaders, criterion, optimizer_vgg16, num_epochs=25)

# Save the models
torch.save(model_resnet18.state_dict(), 'model_resnet18.pth')
torch.save(model_vgg16.state_dict(), 'model_vgg16.pth')
