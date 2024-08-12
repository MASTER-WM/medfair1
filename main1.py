import argparse
import json
import hashlib
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from fairness_metrics import demographic_parity, equalized_odds


def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, choices=['baseline', 'CFair', 'LAFTR', 'resampling'],
                        default='baseline')
    parser.add_argument('--sensitive_name', default='Age_Category', choices=['Age_Category'])
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--total_epochs', type=int, default=25)
    parser.add_argument('--fair_coeff', type=float, default=1.0)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    opt = create_experiment_setting(opt)
    return opt


def create_experiment_setting(opt):
    run_hash = hashlib.sha1()
    run_hash.update(str(time.time()).encode('utf-8'))
    opt['hash'] = run_hash.hexdigest()[:10]
    print('run hash (first 10 digits): ', opt['hash'])

    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')

    return opt


def train_model(model, dataloaders, criterion, optimizer, opt, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if (phase == 'train'):
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, protected_attrs in dataloaders[phase]:
                inputs = inputs.to(opt['device'])
                labels = labels.to(opt['device'])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        if opt['experiment'] == 'CFair':
                            fair_loss = compute_fairness_loss(outputs, protected_attrs, opt['fair_coeff'])
                            loss += fair_loss
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    return model


def evaluate_bias(model, dataloader, opt):
    model.eval()
    all_preds = []
    all_labels = []
    all_protected_attrs = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(opt['device'])
            labels = data[1].to(opt['device'])
            protected_attrs = data[2]

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_protected_attrs.extend(protected_attrs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_protected_attrs = np.array(all_protected_attrs)

    accuracy = np.mean(all_preds == all_labels)
    dp = demographic_parity(all_labels, all_preds, all_protected_attrs)
    eo = equalized_odds(all_labels, all_preds, all_protected_attrs)

    return accuracy, dp, eo


def print_fairness_metrics(model_name, accuracy, dp, eo):
    print(f"\n=== Fairness Metrics for {model_name} ===")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nDemographic Parity:")
    dp_df = pd.DataFrame.from_dict(dp, orient='index', columns=['Demographic Parity'])
    print(dp_df)
    print("\nEqualized Odds:")
    eo_df = pd.DataFrame.from_dict(eo, orient='index')
    print(eo_df)


if __name__ == "__main__":
    opt = collect_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root='path_to_train_data', transform=transform)
    val_dataset = datasets.ImageFolder(root='path_to_val_data', transform=transform)
    test_dataset = datasets.ImageFolder(root='path_to_test_data', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=opt['batch_size'], shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    models_list = [
        models.resnet18(pretrained=True),
        models.vgg16(pretrained=True),
        models.densenet121(pretrained=True),
        models.mobilenet_v2(pretrained=True),
        models.alexnet(pretrained=True)
    ]

    optimizers = [optim.SGD(model.parameters(), lr=opt['lr'], momentum=0.9) for model in models_list]
    model_names = ['ResNet18', 'VGG16', 'DenseNet121', 'MobileNetV2', 'AlexNet']

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    trained_models = []

    for model, optimizer, model_name in zip(models_list, optimizers, model_names):
        model = model.to(opt['device'])
        print(f"Training {model_name} with experiment {opt['experiment']}...")
        trained_model = train_model(model, dataloaders, criterion, optimizer, opt, num_epochs=opt['total_epochs'])
        trained_models.append(trained_model)
        torch.save(trained_model.state_dict(), f'model_{model_name}.pth')
        print(f"{model_name} trained and saved.")

    for model, model_name in zip(trained_models, model_names):
        accuracy, dp, eo = evaluate_bias(model, test_loader, opt)
        print_fairness_metrics(model_name, accuracy, dp, eo)