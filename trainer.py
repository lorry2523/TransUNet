import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, compute_metrics
from utils import DiceLoss
from torchvision import transforms
import random
import functools
from torch.cuda.amp import autocast, GradScaler

#早停验证指标
class EarlyStopping:
    """早停管理，监控验证指标"""
    def __init__(self, patience=15, verbose=False, delta=0.001, path='best_model.pth', mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode
        self.val_score_max = -float('inf') if mode == 'max' else float('inf')

    def __call__(self, val_score, model):
        score = val_score if self.mode == 'max' else -val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f'Validation score ({self.mode}) improved. Saving model ...')
        torch.save(model.state_dict(), self.path)

def _worker_init_fn(seed, worker_id):
    """设置每个 worker 的随机种子，确保可重复性"""
    random.seed(seed + worker_id)
def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    import logging
    import sys
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # 训练集
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    # 验证集
    val_root = args.root_path.replace('train_npz', 'val_npz')
    db_val = Synapse_dataset(base_dir=val_root, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    def validate():
        model.eval()
        miou_sum, dice_sum, pa_sum, cnt = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in valloader:
                img, label = batch['image'].cuda(), batch['label'].cuda()
                with autocast():
                    pred = torch.argmax(torch.softmax(model(img), dim=1), dim=1)
                iou, dice, pa = compute_metrics(pred, label, num_classes)
                miou_sum += iou
                dice_sum += dice
                pa_sum += pa
                cnt += 1
        model.train()
        return (miou_sum / cnt, dice_sum / cnt, pa_sum / cnt) if cnt else (0, 0, 0)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    scaler = GradScaler()
    writer = SummaryWriter(snapshot_path + '/log')
    early_stopping = EarlyStopping(patience=20, verbose=True, mode='max',
                                   path=os.path.join(snapshot_path, 'best_model.pth'))

    iter_num = 0
    max_epoch = args.max_epochs
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            with autocast():
                outputs = model(image_batch)
                loss = 0.5 * ce_loss(outputs, label_batch.long()) + 0.5 * dice_loss(outputs, label_batch, softmax=True)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)

        val_miou, val_dice, val_pa = validate()
        logging.info(f'Epoch {epoch_num}: val mIoU={val_miou:.4f}, Dice={val_dice:.4f}, PA={val_pa:.4f}')
        writer.add_scalar('val/mIoU', val_miou, epoch_num)
        writer.add_scalar('val/Dice', val_dice, epoch_num)
        writer.add_scalar('val/PA', val_pa, epoch_num)

        scheduler.step()
        early_stopping(val_miou, model)
        if early_stopping.early_stop:
            logging.info(f"Early stopped at epoch {epoch_num}")
            break

    writer.close()
    return "Training Finished!"