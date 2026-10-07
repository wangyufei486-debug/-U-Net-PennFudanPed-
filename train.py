import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def set_seed(seed: int):
    """固定模型初始化和数据打乱顺序，使不同超参数实验可以公平比较。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NullExperiment:
    """未安装 W&B 时保留训练流程，但跳过在线实验记录。"""
    def __init__(self):
        self.config = self

    def update(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        optimizer_name: str = 'rmsprop',
        seed: int = 0,
        run_name: str = 'experiment',
        weight_decay=None,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    loader_args = dict(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False
    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_args
    )
    # 小型数据集也应验证最后一个不足 batch 的样本，避免丢失验证数据。
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    if wandb is None:
        logging.warning('Weights & Biases is not installed; continuing without W&B logging')
        experiment = NullExperiment()
    else:
        experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             optimizer=optimizer_name, seed=seed, run_name=run_name,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Optimizer:       {optimizer_name}
        Random seed:     {seed}
        Run name:        {run_name}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'rmsprop':
        effective_weight_decay = 1e-8 if weight_decay is None else weight_decay
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=effective_weight_decay,
            momentum=momentum,
            foreach=True
        )
    elif optimizer_name == 'adamw':
        effective_weight_decay = 1e-4 if weight_decay is None else weight_decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=effective_weight_decay,
            foreach=True
        )
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    logging.info(f'Optimizer weight decay: {effective_weight_decay}')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    # 单独保留“保存最佳模型”的开关，并关闭旧的逐 Epoch 保存逻辑。
    # 旧保存代码仍留在下方以减少对现有训练流程的改动，但条件恒为 False。
    save_best_checkpoint = save_checkpoint
    save_checkpoint = False
    best_val_score = -1.0
    safe_run_name = ''.join(
        char if char.isalnum() or char in ('-', '_', '.') else '_'
        for char in run_name
    )
    best_checkpoint_path = dir_checkpoint / f'{safe_run_name}_best.pth'

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        ce_loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        dice_component = dice_loss(
                            torch.sigmoid(masks_pred.squeeze(1)),
                            true_masks.float(),
                            multiclass=False
                        )
                    else:
                        ce_loss = criterion(masks_pred, true_masks)
                        probabilities = F.softmax(masks_pred, dim=1).float()
                        true_masks_one_hot = F.one_hot(
                            true_masks, model.n_classes
                        ).permute(0, 3, 1, 2).float()
                        # 与验证指标保持一致：背景类别不参与 DiceLoss。
                        dice_component = dice_loss(
                            probabilities[:, 1:],
                            true_masks_one_hot[:, 1:],
                            multiclass=True
                        )
                    loss = ce_loss + dice_component

                if not torch.isfinite(loss):
                    current_lr = optimizer.param_groups[0]['lr']
                    raise FloatingPointError(
                        f'Non-finite loss detected at epoch {epoch}, step {global_step + 1}. '
                        f'CE={float(ce_loss):.6f}, DiceLoss={float(dice_component):.6f}, '
                        f'learning_rate={current_lr}. Try a lower learning rate or disable AMP.'
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{
                    'loss': loss.item(),
                    'CE': ce_loss.item(),
                    'DiceLoss': dice_component.item()
                })

                # Evaluation round
                # 每个 Epoch 结束时验证一次，避免同一轮多次调整学习率。
                division_step = len(train_loader)
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        if wandb is not None:
                            for tag, value in model.named_parameters():
                                tag = tag.replace('/', '.')
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        val_score_value = float(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score_value))

                        # 验证 Dice 越大越好；创新高时覆盖同一个最佳模型文件。
                        if save_best_checkpoint and val_score_value > best_val_score:
                            best_val_score = val_score_value
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            state_dict['mask_values'] = dataset.mask_values
                            torch.save(state_dict, str(best_checkpoint_path))
                            logging.info(
                                f'New best checkpoint saved! '
                                f'Epoch: {epoch}, validation Dice: {best_val_score:.6f}, '
                                f'path: {best_checkpoint_path}'
                            )
                        if wandb is not None:
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except Exception:
                                pass

        logging.info(f'Epoch {epoch} average training loss: {epoch_loss / max(len(train_loader), 1):.6f}')

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--optimizer', choices=['rmsprop', 'adamw'], default='rmsprop',
                        help='Optimizer used for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum used by RMSprop (ignored by AdamW)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Optimizer weight decay; defaults to 1e-8 for RMSprop and 1e-4 for AdamW')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed used for initialization, split and shuffling')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name used for the best checkpoint file')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    set_seed(args.seed)
    run_name = args.run_name or f'{args.optimizer}_lr{args.lr:g}_seed{args.seed}'

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            optimizer_name=args.optimizer,
            seed=args.seed,
            run_name=run_name,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            optimizer_name=args.optimizer,
            seed=args.seed,
            run_name=run_name,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
