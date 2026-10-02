import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from evaluate import evaluate
from unet import UNet
from utils.data_loading import CarvanaDataset


def get_args():
    parser = argparse.ArgumentParser(description='Validate a trained U-Net checkpoint')
    parser.add_argument('--model', '-m', type=Path, required=True, help='Checkpoint path')
    parser.add_argument('--images', type=Path, default=Path('./data/imgs'))
    parser.add_argument('--masks', type=Path, default=Path('./data/masks'))
    parser.add_argument('--batch-size', '-b', type=int, default=2)
    parser.add_argument('--scale', '-s', type=float, default=0.5)
    parser.add_argument('--validation', '-v', type=float, default=10.0,
                        help='Validation percentage used by train.py')
    parser.add_argument('--classes', '-c', type=int, default=2)
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = get_args()
    if not args.model.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.model}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CarvanaDataset(args.images, args.masks, args.scale)
    n_val = int(len(dataset) * args.validation / 100)
    n_train = len(dataset) - n_val
    _, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', None)
    model.load_state_dict(state_dict)
    model.to(device=device, memory_format=torch.channels_last)

    # 使用与 train.py 完全相同的划分和 Dice 逻辑验证最终 checkpoint。
    score = evaluate(model, val_loader, device, args.amp)
    print(f'Device: {device}')
    print(f'Validation samples: {n_val}')
    print(f'Mask values: {mask_values}')
    print(f'Validation Dice: {score.item():.6f}')


if __name__ == '__main__':
    main()
