import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import argparse
import hashlib
import time
import os
import json


# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx]['image']
        image = Image.open(image).convert('RGB')
        label = self.dataframe.iloc[idx]['binaryLabel']
        protected_attr = self.dataframe.iloc[idx]['Age_Category']

        if self.transform:
            image = self.transform(image)

        return image, label, protected_attr


# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Arguments collection
def collect_args():
    parser = argparse.ArgumentParser()

    # experiments
    parser.add_argument('--experiment', type=str, choices=[
        'baseline', 'CFair', 'LAFTR', 'LNL', 'EnD', 'DomainInd', 'resampling',
        'ODR', 'SWA', 'SWAD', 'SAM', 'GSAM', 'SAMSWAD', 'GroupDRO', 'BayesCNN', 'resamplingSWAD'
    ])

    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--wandb_name', type=str, default='baseline')
    parser.add_argument('--if_wandb', type=bool, default=True)
    parser.add_argument('--dataset_name', default='CXP', choices=[
        'CXP', 'NIH', 'MIMIC_CXR', 'RadFusion', 'RadFusion4', 'HAM10000', 'HAM100004',
        'Fitz17k', 'OCT', 'PAPILA', 'ADNI', 'ADNI3T', 'COVID_CT_MD', 'RadFusion_EHR', 'MIMIC_III', 'eICU'
    ])

    parser.add_argument('--resume_path', type=str, default='', help='explicitly identify checkpoint path to resume.')
    parser.add_argument('--sensitive_name', default='Sex', choices=['Sex', 'Age', 'Race', 'skin_type', 'Insurance'])
    parser.add_argument('--is_3d', type=bool, default=False)
    parser.add_argument('--is_tabular', type=bool, default=False)

    # training
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate of the learning rate')
    parser.add_argument('--lr_decay_period', type=float, default=10, help='decay period of the learning rate')
    parser.add_argument('--total_epochs', type=int, default=15, help='total training epochs')
    parser.add_argument('--early_stopping', type=int, default=5, help='early stopping epochs')
    parser.add_argument('--test_mode', type=bool, default=False, help='if using test mode')
    parser.add_argument('--hyper_search', type=bool, default=False, help='if searching hyper-parameters')

    # testing
    parser.add_argument('--hash_id', type=str, default='')

    # strategy for validation
    parser.add_argument('--val_strategy', type=str, default='loss', choices=['loss', 'worst_auc'], help='strategy for selecting val model')

    # cross-domain
    parser.add_argument('--cross_testing', action='store_true')
    parser.add_argument('--source_domain', default='', choices=['CXP', 'MIMIC_CXR', 'ADNI', 'ADNI3T'])
    parser.add_argument('--target_domain', default='', choices=['CXP', 'MIMIC_CXR', 'ADNI', 'ADNI3T'])
    parser.add_argument('--cross_testing_model_path', type=str, default='', help='path of the models of three random seeds')
    parser.add_argument('--cross_testing_model_path_single', type=str, default='', help='path of the models')

    # network
    parser.add_argument('--backbone', default='cusResNet18', choices=[
        'cusResNet18', 'cusResNet50', 'cusDenseNet121', 'cusResNet18_3d', 'cusResNet50_3d', 'cusMLP'
    ])
    parser.add_argument('--pretrained', type=bool, default=True, help='if use pretrained ResNet backbone')
    parser.add_argument('--output_dim', type=int, default=14, help='output dimension of the classification network')
    parser.add_argument('--num_classes', type=int, default=14, help='number of target classes')
    parser.add_argument('--sens_classes', type=int, default=2, help='number of sensitive classes')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel of the images')

    # resampling
    parser.add_argument('--resample_which', type=str, default='group', choices=['class', 'balanced'], help='audit step for LAFTR')

    # LAFTR
    parser.add_argument('--aud_steps', type=int, default=1, help='audit step for LAFTR')
    parser.add_argument('--class_coeff', type=float, default=1.0, help='coefficient for classification loss of LAFTR')
    parser.add_argument('--fair_coeff', type=float, default=1.0, help='coefficient for fair loss of LAFTR')
    parser.add_argument('--model_var', type=str, default='laftr-eqodd', help='model variation for LAFTR')

    # CFair
    parser.add_argument('--mu', type=float, default=0.1, help='coefficient for adversarial loss of CFair')

    # LNL
    parser.add_argument('--_lambda', type=float, default=0.1, help='coefficient for loss of LNL')

    # EnD
    parser.add_argument('--alpha', type=float, default=0.1, help='weighting parameters alpha for EnD method')
    parser.add_argument('--beta', type=float, default=0.1, help='weighting parameters beta for EnD method')

    # ODR
    parser.add_argument("--lambda_e", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--lambda_od", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--gamma_e", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--gamma_od", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--step_size", type=int, default=20, help="step size for adjusting coefficients for loss of ODR")

    # GroupDRO
    parser.add_argument("--groupdro_alpha", type=float, default=0.2, help="coefficient alpha for loss of GroupDRO")
    parser.add_argument("--groupdro_gamma", type=float, default=0.1, help="coefficient gamma for loss of GroupDRO")

    # SWA
    parser.add_argument("--swa_start", type=int, default=7, help="starting epoch for averaging of SWA")
    parser.add_argument("--swa_lr", type=float, default=0.0001, help="learning rate for averaging of SWA")
    parser.add_argument("--swa_annealing_epochs", type=int, default=3, help="learning rate for averaging of SWA")

    # SWAD
    parser.add_argument("--swad_n_converge", type=int, default=3, help="starting converging epoch of SWAD")
    parser.add_argument("--swad_n_tolerance", type=int, default=6, help="tolerance steps of SWAD")
    parser.add_argument("--swad_tolerance_ratio", type=float, default=0.05, help="tolerance ratio of SWAD")
    parser.add_argument("--swad_end", type=int, default=14, help="ending epoch for averaging of SWAD")

    # SAM
    parser.add_argument("--rho", type=float, default=2, help="Rho parameter for SAM.")
    parser.add_argument("--adaptive", type=bool, default=True, help="whether using adaptive mode for SAM.")
    parser.add_argument("--T_max", type=int, default=50, help="Value for LR scheduler")

    # GSAM
    parser.add_argument("--gsam_alpha", type=float, default=2, help="Rho parameter for SAM.")

    # BayesCNN
    parser.add_argument("--num_monte_carlo", type=int, default=10, help="Rho parameter for SAM.")

    parser.set_defaults(cuda=True)

    # logging
    parser.add_argument('--log_freq', type=int, default=50, help='logging frequency (step)')

    opt = vars(parser.parse_args())
    opt = create_experiment_setting(opt)
    return opt

def create_experiment_setting(opt):
    run_hash = hashlib.sha1()
    run_hash.update(str(time.time()).encode('utf-8'))
    opt['hash'] = run_hash.hexdigest()[:10]
    print('run hash (first 10 digits): ', opt['hash'])
    opt['device'] = torch.device('cuda' if torch.cuda.is_available() and opt['cuda'] else 'cpu')

    # Create the necessary directories
    opt['log_dir'] = os.path.join('./logs', opt['experiment_name'], opt['hash'])
    os.makedirs(opt['log_dir'], exist_ok=True)

    with open(os.path.join(opt['log_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f, indent=4, sort_keys=True)

    return opt


if __name__ == "__main__":
    args = collect_args()
    print(args)

# Function to evaluate model performance and bias
def evaluate_bias(model, test_loader, protected_attr_name):
    model.eval()
    all_labels = []
    all_preds = []
    all_protected_attrs = []

    with torch.no_grad():
        for images, labels, protected_attrs in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_protected_attrs.extend(protected_attrs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    df = pd.DataFrame({
        'label': all_labels,
        'pred': all_preds,
        protected_attr_name: all_protected_attrs
    })

    # Calculate accuracy and F1-Score for each protected group
    age_group_accuracies = {}
    age_group_f1_scores = {}
    for group in df[protected_attr_name].unique():
        group_df = df[df[protected_attr_name] == group]
        group_accuracy = accuracy_score(group_df['label'], group_df['pred'])
        group_f1 = f1_score(group_df['label'], group_df['pred'], average='weighted')
        age_group_accuracies[group] = group_accuracy
        age_group_f1_scores[group] = group_f1

    # Calculate demographic parity and equalized odds
    dp = {}
    eo = {}
    for group in df[protected_attr_name].unique():
        group_df = df[df[protected_attr_name] == group]
        dp[group] = group_df['pred'].mean()
        eo[group] = group_df['pred'].std()

    return accuracy, f1, age_group_accuracies, age_group_f1_scores, dp, eo


# Function to print fairness metrics
def print_fairness_metrics(model_name, accuracy, f1, age_group_accuracies, age_group_f1_scores, dp, eo):
    print(f"\n=== Fairness Metrics for {model_name} ===")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall F1-Score: {f1:.4f}")

    print("\nAccuracy by Age Group:")
    for age_group, group_accuracy in age_group_accuracies.items():
        print(f"  Age group {age_group}: {group_accuracy:.4f}")

    print("\nF1-Score by Age Group:")
    for age_group, group_f1 in age_group_f1_scores.items():
        print(f"  Age group {age_group}: {group_f1:.4f}")

    print("\nDemographic Parity:")
    dp_df = pd.DataFrame.from_dict(dp, orient='index', columns=['Demographic Parity'])
    print(dp_df)

    print("\nEqualized Odds:")
    eo_df = pd.DataFrame.from_dict(eo, orient='index')
    print(eo_df)


# Main function to train and evaluate models
def main():
    args = collect_args()

    # Load and preprocess the data
    train_df = pd.read_csv('path_to_train_data.csv')
    test_df = pd.read_csv('path_to_test_data.csv')

    train_dataset = CustomDataset(train_df, transform=transform)
    test_dataset = CustomDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define models (example with ResNet18)
    if args.backbone == 'cusResNet18':
        model = models.resnet18(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.backbone == 'cusResNet50':
        model = models.resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    if torch.cuda.is_available():
        model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_accuracy = 0
    early_stopping_counter = 0

    for epoch in range(args.total_epochs):
        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{args.total_epochs}], Loss: {epoch_loss:.4f}")

        # Early stopping
        accuracy, _, _, _, _, _ = evaluate_bias(model, test_loader, protected_attr_name=args.sensitive_name)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stopping_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.early_stopping:
                print("Early stopping triggered")
                break

    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    accuracy, f1, age_group_accuracies, age_group_f1_scores, dp, eo = evaluate_bias(model, test_loader,
                                                                                    protected_attr_name=args.sensitive_name)
    print_fairness_metrics(args.backbone, accuracy, f1, age_group_accuracies, age_group_f1_scores, dp, eo)


if __name__ == "__main__":
    main()