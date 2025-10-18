import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
import networks.vit_seg_modeling
from trainer import trainer_breast_ultrasound
import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/BreastUltrasound/', help='root dir for data')
    parser.add_argument('--dataset_name', type=str,
                        default='BreastUltrasound', help='dataset name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_breast_ultrasound', help='list dir')
    parser.add_argument('--model_path', type=str,
                        default='../model', help='path for the model')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--save_per_n_epochs', type=int,
                        default=20, help='save the model at every n epoch')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    return parser.parse_args()


def set_framework_settings(args):
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_dataset_by_name(dataset_name):
    dataset_config = {
        'BreastUltrasound': {
            'name': dataset_name,
            'root_path': '../data/BreastUltrasound/',
            'list_dir': './lists/lists_breast_ultrasound/',
            'num_classes': 3,
        },
    }
    return dataset_config[dataset_name]


def extend_args_with_dataset(args, dataset):
    args.num_classes = dataset['num_classes']
    args.root_path = dataset['root_path']
    args.list_dir = dataset['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset['name'] + str(args.img_size)


def get_vit_config(args):
    vit_config = networks.vit_seg_modeling.CONFIGS[args.vit_name]
    vit_config.n_classes = args.num_classes
    vit_config.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        vit_config.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    return vit_config


def load_network(snapshot_path, args, vit_config):
    network = ViT_seg(vit_config, img_size=args.img_size, num_classes=vit_config.n_classes).cuda()

    if os.path.exists(snapshot_path):
        checkpoint_files = sorted(
            [f for f in os.listdir(snapshot_path) if f.startswith('epoch_') and f.endswith('.pth')]
        )
    else:
        os.makedirs(snapshot_path)
        checkpoint_files = []

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        latest_path = os.path.join(snapshot_path, latest_checkpoint)
        print(f"Found checkpoint: {latest_path} — resuming training.")
        checkpoint = torch.load(latest_path)
        network.load_state_dict(checkpoint['model'])
        starting_epoch = checkpoint['epoch']
    else:
        print("No checkpoints found in directory — starting from pretrained weights.")
        network.load_from(weights=np.load(vit_config.pretrained_path))
        starting_epoch = 0
    return network, starting_epoch


def main():
    args = get_args()
    set_framework_settings(args)
    dataset = get_dataset_by_name(args.dataset_name)
    extend_args_with_dataset(args, dataset)

    snapshot_path = utils.get_snapshot_path(args)

    vit_config = get_vit_config(args)

    network, starting_epoch = load_network(snapshot_path, args, vit_config)

    trainer = {'BreastUltrasound': trainer_breast_ultrasound}
    trainer[dataset['name']](args, network, snapshot_path, starting_epoch)


if __name__ == "__main__":
    main()
