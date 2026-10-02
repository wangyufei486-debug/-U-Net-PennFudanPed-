import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Prepare PennFudanPed for binary U-Net segmentation')
    parser.add_argument('--source', type=Path, default=Path('./PennFudanPed'),
                        help='PennFudanPed root containing PNGImages and PedMasks')
    parser.add_argument('--output', type=Path, default=Path('./data'),
                        help='Output root; images and masks are written to imgs and masks')
    parser.add_argument('--size', type=int, default=512,
                        help='Square canvas size before train.py applies its scale factor')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite already prepared files with the same names')
    return parser.parse_args()


def letterbox(pil_img, size, is_mask):
    """等比例缩放并补边，保证 DataLoader 能将不同尺寸的图片组成 batch。"""
    width, height = pil_img.size
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    resized = pil_img.resize((new_width, new_height), resample=resample)

    canvas = Image.new('L' if is_mask else 'RGB', (size, size), color=0)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def prepare_dataset(source, output, size, overwrite):
    image_dir = source / 'PNGImages'
    mask_dir = source / 'PedMasks'
    output_images = output / 'imgs'
    output_masks = output / 'masks'

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f'Expected {image_dir} and {mask_dir}. '
            'Download and extract PennFudanPed before running this script.'
        )
    if size <= 0:
        raise ValueError('--size must be greater than 0')

    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob('*.png'))
    mask_files = sorted(mask_dir.glob('*_mask.png'))
    image_ids = {path.stem for path in image_files}
    mask_ids = {path.stem.removesuffix('_mask') for path in mask_files}
    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)

    if missing_masks or missing_images:
        print(f'Images: {len(image_files)}')
        print(f'Masks: {len(mask_files)}')
        print(f'Images without masks: {missing_masks}')
        print(f'Masks without images: {missing_images}')
        raise RuntimeError('Image/mask matching failed; no output was written')
    if not image_files:
        raise RuntimeError(f'No PNG images found in {image_dir}')

    existing = []
    for image_path in image_files:
        output_image = output_images / image_path.name
        output_mask = output_masks / f'{image_path.stem}_mask.png'
        if output_image.exists() or output_mask.exists():
            existing.extend(path for path in (output_image, output_mask) if path.exists())
    if existing and not overwrite:
        preview = ', '.join(str(path) for path in existing[:3])
        raise FileExistsError(f'Prepared files already exist ({preview}). Use --overwrite to replace them.')

    processed = 0
    for image_path in tqdm(image_files, desc='Preparing PennFudanPed', unit='pair'):
        mask_path = mask_dir / f'{image_path.stem}_mask.png'
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            image = image_file.convert('RGB')
            mask_array = np.asarray(mask_file)
            if mask_array.ndim != 2:
                raise ValueError(f'Mask must be single-channel: {mask_path} has shape {mask_array.shape}')
            if image.size != mask_file.size:
                raise ValueError(
                    f'Image/mask size mismatch for {image_path.stem}: '
                    f'{image.size} vs {mask_file.size}'
                )

            # PennFudan 的实例编号全部合并为 person 类：0=背景，255=人物。
            binary_mask = Image.fromarray(np.where(mask_array > 0, 255, 0).astype(np.uint8), mode='L')
            prepared_image = letterbox(image, size, is_mask=False)
            prepared_mask = letterbox(binary_mask, size, is_mask=True)

            prepared_image.save(output_images / image_path.name)
            prepared_mask.save(output_masks / mask_path.name)
            processed += 1

    print(f'Images: {len(image_files)}')
    print(f'Masks: {len(mask_files)}')
    print(f'Processed pairs: {processed}')
    print(f'Image output: {output_images.resolve()}')
    print(f'Mask output: {output_masks.resolve()}')
    print('Matching failures: 0')
    print('Mask values: 0=background, 255=person')
    print(f'Prepared size: {size}x{size}')


if __name__ == '__main__':
    args = get_args()
    prepare_dataset(args.source, args.output, args.size, args.overwrite)
