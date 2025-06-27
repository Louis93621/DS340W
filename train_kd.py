import argparse
import sys
import os
from pathlib import Path
import torch
import yaml
from copy import deepcopy
from models.yolo import Model
from models.experimental import attempt_load
from utils.general import (check_img_size, check_dataset, yaml_save, increment_path)
from utils.torch_utils import select_device, ModelEMA
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.loss_tal import ComputeLoss  # student base loss
from utils.metrics import fitness
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import csv
# ======================
# ComputeLossKD
# ======================
class ComputeLossKD(ComputeLoss):
    def __init__(self, model, teacher_model, lambda_kd):
        super().__init__(model)
        self.teacher_model = teacher_model
        self.lambda_kd = lambda_kd
        self.kd_loss_fn = nn.MSELoss()

    def __call__(self, preds, targets, imgs=None):
        student_pred = preds[0] if isinstance(preds, (list, tuple)) else preds
        loss, loss_items = super().__call__(student_pred, targets)

        # teacher prediction
        with torch.no_grad():
            teacher_preds = self.teacher_model(imgs)
            teacher_pred = teacher_preds[0] if isinstance(teacher_preds, (list, tuple)) else teacher_preds

        # distill classification scores only
        kd_loss = 0.
        for sp, tp in zip(student_pred, teacher_pred):
            # sp, tp shape: [B, anchors, 4 + num_cls + dfl]
            sp_cls = sp[..., 4:4 + self.nc]
            tp_cls = tp[..., 4:4 + self.nc]

            # make sure they have the same shape
            if sp_cls.shape == tp_cls.shape:
                kd_loss += self.kd_loss_fn(sp_cls, tp_cls)

        loss += self.lambda_kd * kd_loss
        return loss, loss_items


# ======================
# Train loop
# ======================
def increment_dir(dir_path):
    """
    如果資料夾已存在，為資料夾名加上數字後綴，直到找到唯一的名稱。
    """
    counter = 1
    new_dir_path = dir_path
    while new_dir_path.exists():  # 檢查資料夾是否存在
        new_dir_path = dir_path.with_name(f"{dir_path.name}_{counter}")  # 新的名稱
        counter += 1
    return new_dir_path




def train(hyp, opt, device, callbacks):
    save_dir = Path(opt.save_dir)
    save_dir = increment_dir(save_dir)  # 確保資料夾唯一
    save_dir.mkdir(parents=True, exist_ok=True)  # 創建資料夾，若已存在則跳過

    # 其餘訓練程式碼...
    print(f"訓練結果將儲存在：{save_dir}")
    epochs, batch_size = opt.epochs, opt.batch_size

    # Load data.yaml
    data_dict = check_dataset(opt.data)
    nc = int(data_dict['nc'])
    names = data_dict['names']

    # Load student model from YAML only
    model = Model(opt.cfg, ch=3, nc=nc).to(device)

    # Load teacher model from pre-trained .pt
    teacher_model = attempt_load(opt.teacher_weights, device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Add .hyp to student
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)
    model.hyp = hyp

    # Create dataloader
    imgsz = check_img_size(opt.imgsz, max(int(model.stride.max()), 32))
    train_loader, _ = create_dataloader(data_dict['train'], imgsz, batch_size, int(model.stride.max()),
                                        hyp=hyp, augment=True, cache=None, rect=False, rank=-1, workers=8, prefix='train: ')
    val_loader = create_dataloader(data_dict['val'], imgsz, batch_size*2, int(model.stride.max()),
                                   hyp=hyp, cache=None, rect=True, rank=-1, workers=8, prefix='val: ')[0]

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - x/epochs)**hyp['lrf'])

    # Loss function
    compute_loss = ComputeLossKD(model, teacher_model, opt.lambda_kd)

    # EMA
    ema = ModelEMA(model)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    nb = len(train_loader)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        mloss = torch.zeros(3, device=device)
        pbar = tqdm(enumerate(train_loader), total=nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device).float() / 255
            targets = targets.to(device)

            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss, loss_items = compute_loss(preds, targets, imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            ema.update(model)

            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {mloss}")

        scheduler.step()
        
        # Save checkpoint at each epoch
        ckpt_name = save_dir / f'epoch_{epoch+1}.pt'
        torch.save(ema.ema.state_dict(), ckpt_name)
        print(f"✅ Saved checkpoint: {ckpt_name}")

    # Save final
    torch.save(ema.ema.state_dict(), save_dir / 'student_final.pt')
    print(f"✅ Final model saved to {save_dir / 'student_final.pt'}")

# ======================
# Parse options
# ======================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-weights', type=str, required=True)
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--hyp', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lambda_kd', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='runs/train_kd')
    return parser.parse_args()

# ======================
# Main
# ======================
def main(opt):
    device = select_device(opt.device)
    train(opt.hyp, opt, device, Callbacks())

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = increment_path(opt.save_dir)
    main(opt)

