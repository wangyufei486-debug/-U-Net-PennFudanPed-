import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.data_loading import CarvanaDataset


def get_args():
    parser = argparse.ArgumentParser(description='Check prepared segmentation images and masks')
    parser.add_argument('--images', type=Path, default=Path('./data/imgs'))
    parser.add_argument('--masks', type=Path, default=Path('./data/masks'))
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--samples', type=int, default=3)
    return parser.parse_args()


def main():
    args = get_args()
    image_files = sorted(path for path in args.images.glob('*') if path.is_file() and not path.name.startswith('.'))
    mask_files = sorted(path for path in args.masks.glob('*') if path.is_file() and not path.name.startswith('.'))
    image_ids = {path.stem for path in image_files}
    mask_ids = {path.stem.removesuffix('_mask') for path in mask_files}

    print(f'Images: {len(image_files)}')
    print(f'Masks: {len(mask_files)}')
    print(f'Images without masks: {sorted(image_ids - mask_ids)}')
    print(f'Masks without images: {sorted(mask_ids - image_ids)}')
    if image_ids != mask_ids:
        raise RuntimeError('Image/mask matching failed')

    dataset = CarvanaDataset(args.images, args.masks, args.scale)
    rng = random.Random(0)
    indices = rng.sample(range(len(dataset)), k=min(args.samples, len(dataset)))
    for index in indices:
        sample = dataset[index]
        image = sample['image']
        mask = sample['mask']
        print(f'\nSample index: {index}')
        print(f'image shape: {image.shape}')
        print(f'image dtype: {image.dtype}')
        print(f'mask shape: {mask.shape}')
        print(f'mask dtype: {mask.dtype}')
        print(f'mask unique: {torch.unique(mask)}')

    # Windows 下使用 0 个 worker 做快速预检，避免检查阶段额外创建子进程。
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print('\nDataLoader batch check: OK')
    print(f"batch image shape: {batch['image'].shape}")
    print(f"batch mask shape: {batch['mask'].shape}")
    print(f"batch mask unique: {torch.unique(batch['mask'])}")


if __name__ == '__main__':
    main()
