import os
import argparse
from models.basemodels_mlp import cusMLP
import torch
import models
from utils import basics
import wandb
import json
import hashlib
import time

def collect_args():
    parser = argparse.ArgumentParser()

    # Experiments
    parser.add_argument('--experiment',
                        type=str,
                        choices=[
                            'baseline',
                            'CFair',
                            'LAFTR',
                            'LNL',
                            'EnD',
                            'DomainInd',
                            'resampling',
                            'ODR',
                            'SWA',
                            'SWAD',
                            'SAM',
                            'GSAM',
                            'SAMSWAD',
                            'GroupDRO',
                            'BayesCNN',
                            'resamplingSWAD',
                        ],
                        default='baseline')

    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--wandb_name', type=str, default='baseline')
    parser.add_argument('--if_wandb', type=bool, default=True)

    # Dataset and sensitive attributes
    parser.add_argument('--dataset_name', default='CXP', choices=[
                        'CXP', 'NIH', 'MIMIC_CXR', 'RadFusion', 'RadFusion4', 
                        'HAM10000', 'HAM100004', 'Fitz17k', 'OCT', 'PAPILA', 'ADNI', 
                        'ADNI3T', 'COVID_CT_MD','RadFusion_EHR', 'MIMIC_III', 'eICU'])

    parser.add_argument('--sensitive_name', default='Sex', choices=[
                        'Sex', 'Age', 'Race', 'skin_type', 'Insurance'])

    parser.add_argument('--is_3d', type=bool, default=False)
    parser.add_argument('--is_tabular', type=bool, default=False)

    # Training settings
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
    parser.add_argument('--resume_path', type=str, default='', help='path to resume the training from a checkpoint')
    parser.add_argument('--data_setting', type=dict, default={'augment': True, 'other_key': 'value'},
                        help='Setting for data configuration')

    # Testing settings
    parser.add_argument('--hash_id', type=str, default='')

    # Strategy for validation
    parser.add_argument('--val_strategy', type=str, default='loss', choices=['loss', 'worst_auc'], help='strategy for selecting val model')

    # Cross-domain settings
    parser.add_argument('--cross_testing', action='store_true')
    parser.add_argument('--source_domain', default='', choices=['CXP', 'MIMIC_CXR', 'ADNI', 'ADNI3T'])
    parser.add_argument('--target_domain', default='', choices=['CXP', 'MIMIC_CXR', 'ADNI', 'ADNI3T'])
    parser.add_argument('--cross_testing_model_path', type=str, default='', help='path of the models of three random seeds')
    parser.add_argument('--cross_testing_model_path_single', type=str, default='', help='path of the models')

    # Network settings
    parser.add_argument('--backbone', default='cusResNet18', choices=['cusResNet18', 'cusResNet50','cusDenseNet121',
                                'cusResNet18_3d', 'cusResNet50_3d', 'cusMLP'])
    parser.add_argument('--pretrained', type=bool, default=True, help='if use pretrained ResNet backbone')
    parser.add_argument('--output_dim', type=int, default=14, help='output dimension of the classification network')
    parser.add_argument('--num_classes', type=int, default=14, help='number of target classes')
    parser.add_argument('--sens_classes', type=int, default=2, help='number of sensitive classes')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel of the images')

    # Resampling
    parser.add_argument('--resample_which', type=str, default='group', choices=['class', 'balanced'], help='audit step for LAFTR')

    # LAFTR settings
    parser.add_argument('--aud_steps', type=int, default=1, help='audit step for LAFTR')
    parser.add_argument('--class_coeff', type=float, default=1.0, help='coefficient for classification loss of LAFTR')
    parser.add_argument('--fair_coeff', type=float, default=1.0, help='coefficient for fair loss of LAFTR')
    parser.add_argument('--model_var', type=str, default='laftr-eqodd', help='model variation for LAFTR')

    # CFair settings
    parser.add_argument('--mu', type=float, default=0.1, help='coefficient for adversarial loss of CFair')

    # LNL settings
    parser.add_argument('--_lambda', type=float, default=0.1, help='coefficient for loss of LNL')

    # EnD settings
    parser.add_argument('--alpha', type=float, default=0.1, help='weighting parameters alpha for EnD method')
    parser.add_argument('--beta', type=float, default=0.1, help='weighting parameters beta for EnD method')

    # ODR settings
    parser.add_argument("--lambda_e", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--lambda_od", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--gamma_e", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--gamma_od", type=float, default=0.1, help="coefficient for loss of ODR")
    parser.add_argument("--step_size", type=int, default=20, help="step size for adjusting coefficients for loss of ODR")

    # GroupDRO settings
    parser.add_argument("--groupdro_alpha", type=float, default=0.2, help="coefficient alpha for loss of GroupDRO")
    parser.add_argument("--groupdro_gamma", type=float, default=0.1, help="coefficient gamma for loss of GroupDRO")

    # SWA settings
    parser.add_argument("--swa_start", type=int, default=7, help="starting epoch for averaging of SWA")
    parser.add_argument("--swa_lr", type=float, default=0.0001, help="learning rate for averaging of SWA")
    parser.add_argument("--swa_annealing_epochs", type=int, default=3, help="learning rate for averaging of SWA")

    # SWAD settings
    parser.add_argument("--swad_n_converge", type=int, default=3, help="starting converging epoch of SWAD")
    parser.add_argument("--swad_n_tolerance", type=int, default=6, help="tolerance steps of SWAD")
    parser.add_argument("--swad_tolerance_ratio", type=float, default=0.05, help="tolerance ratio of SWAD")

    # SAM settings
    parser.add_argument("--rho", type=float, default=2, help="Rho parameter for SAM.")
    parser.add_argument("--adaptive", type=bool, default=True, help="whether using adaptive mode for SAM.")
    parser.add_argument("--T_max", type=int, default=50, help="Value for LR scheduler")

    # GSAM settings
    parser.add_argument("--gsam_alpha", type=float, default=2, help="Rho parameter for SAM.")

    # BayesCNN settings
    parser.add_argument("--num_monte_carlo", type=int, default=10, help="Rho parameter for SAM.")

    parser.set_defaults(cuda=True)

    # Logging
    parser.add_argument('--log_freq', type=int, default=50, help='logging frequency (step)')

    opt = vars(parser.parse_args())

    opt = create_exerpiment_setting(opt)

    return opt, wandb

def create_exerpiment_setting(opt):
    # get hash
    run_hash = hashlib.sha1()
    run_hash.update(str(time.time()).encode('utf-8'))
    opt['hash'] = run_hash.hexdigest()[:10]
    print('run hash (first 10 digits): ', opt['hash'])

    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')

    opt['save_folder'] = os.path.join('your_path/fariness_data/model_records', 
                                      opt['dataset_name'],
                                      opt['sensitive_name'],
                                      opt['backbone'],
                                      opt['experiment'])

    # ایجاد دایرکتوری برای ذخیره‌سازی اگر وجود ندارد
    if not os.path.exists(opt['save_folder']):
        os.makedirs(opt['save_folder'])

    # تنظیمات wandb
    if opt['if_wandb']:
        wandb.init(project=opt['experiment_name'], name=opt['wandb_name'], config=opt)
        wandb.run.name = f"{opt['dataset_name']}_{opt['sensitive_name']}_{opt['experiment']}_{opt['hash']}"

    return opt