import argparse
import sys
import os
from pathlib import Path
import torch
import yaml
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
import csv
import time

# 確保 utils 和 models 可以被導入
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 專案根目錄
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# 導入必要的模組
import val as validate
from models.yolo import Model
from models.experimental import attempt_load
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_dataset, check_img_size,
                        increment_path, yaml_save, strip_optimizer, one_cycle, intersect_dicts)
from utils.torch_utils import select_device, ModelEMA, de_parallel
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.loss_tal import ComputeLoss as ComputeLossBase
from utils.metrics import fitness
from utils.plots import plot_results


# ===================================================================
# 知識蒸餾損失函數 (ComputeLossKD) - 支援訓練與驗證模式的最終版
# ===================================================================
class ComputeLossKD(ComputeLossBase):
    def __init__(self, model, teacher_model, hyp, lambda_kd=1.0, device=''):
        super().__init__(model)
        
        self.teacher_model = teacher_model
        self.lambda_kd = lambda_kd
        self.device = device
        self.hyp = hyp
        
        self.T = hyp.get('kd_temperature', 3.0)
        self.kd_start_epoch = hyp.get('kd_start_epoch', 3)
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        LOGGER.info(f"ComputeLossKD (Classification Distribution Distill) initialized. Lambda={self.lambda_kd}, Temp={self.T}, KD starts at epoch {self.kd_start_epoch}")

    def __call__(self, p, targets, imgs=None, epoch=0):
        student_p_list = p[0] if isinstance(p, (list, tuple)) and len(p) > 0 and isinstance(p[0], list) else p
        base_loss, base_loss_items = super().__call__(student_p_list, targets)

        if imgs is None or epoch < self.kd_start_epoch:
            if imgs is None: # Validation mode
                return base_loss, base_loss_items
            else: # Training mode, but before KD starts
                all_loss_items = torch.cat((base_loss_items, torch.tensor([0.0, 0.0, 0.0], device=self.device).detach()))
                return base_loss, all_loss_items
        
        with torch.no_grad():
            teacher_p_list = self.teacher_model(imgs)
            teacher_p_list = teacher_p_list[0] if isinstance(teacher_p_list, (list, tuple)) and len(teacher_p_list) > 0 and isinstance(teacher_p_list[0], list) else teacher_p_list

        student_scores_all = [s[..., -self.nc:].view(s.shape[0], -1, self.nc) for s in student_p_list]
        student_agg_scores = torch.cat(student_scores_all, dim=1)

        teacher_scores_all = [pt[..., -self.nc:].view(pt.shape[0], -1, self.nc) for pt in teacher_p_list]
        teacher_agg_scores = torch.cat(teacher_scores_all, dim=1)

        student_dist = torch.mean(torch.nn.functional.softmax(student_agg_scores / self.T, dim=-1), dim=1)
        teacher_dist = torch.mean(torch.nn.functional.softmax(teacher_agg_scores / self.T, dim=-1), dim=1)
        
        kd_loss = self.kd_loss_fn(torch.log(student_dist + 1e-9), teacher_dist) * (self.T * self.T)
        
        total_loss = base_loss + self.lambda_kd * kd_loss
        all_loss_items = torch.cat((base_loss_items, torch.tensor([0.0, kd_loss, 0.0], device=self.device).detach()))
        
        return total_loss, all_loss_items


# ======================================
# 完整訓練與驗證流程 (Train loop)
# ======================================
def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size = Path(opt.save_dir), opt.epochs, opt.batch_size
    w = save_dir / 'weights'; w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'
    
    LOGGER.info(f"Training results will be saved to: {save_dir}")

    if isinstance(hyp, str):
        with open(hyp) as f: hyp = yaml.safe_load(f)
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    data_dict = check_dataset(opt.data)
    nc = int(data_dict['nc'])

    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    model.hyp = hyp

    if opt.student_weights:
        LOGGER.info(f"Loading pretrained weights for student model from {opt.student_weights}...")
        ckpt = torch.load(opt.student_weights, map_location=device)
        state_dict = None
        if isinstance(ckpt, dict):
            model_or_state = ckpt.get('ema') or ckpt.get('model')
            if model_or_state is not None:
                try: state_dict = model_or_state.state_dict()
                except AttributeError: state_dict = model_or_state
            else: state_dict = ckpt
        else: state_dict = ckpt
        if isinstance(state_dict, dict) and state_dict:
            state_dict = intersect_dicts(model.state_dict(), state_dict, exclude=['anchor'])
            model.load_state_dict(state_dict, strict=False)
            LOGGER.info(f'Transferred {len(state_dict)} weights from {opt.student_weights} to student model.')
        else: LOGGER.error(f"Failed to parse a valid state_dict from {opt.student_weights}.")

    teacher_model = attempt_load(opt.teacher_weights, device=device)
    teacher_model.eval()
    for p in teacher_model.parameters(): p.requires_grad = False
    LOGGER.info(f"Loading teacher model from {opt.teacher_weights}...")
    
    imgsz = check_img_size(opt.imgsz, max(int(model.stride.max()), 32))
    train_loader, dataset = create_dataloader(
        data_dict['train'], imgsz, batch_size, int(model.stride.max()),
        hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=-1, workers=opt.workers, prefix='train: ')
    
    val_loader = create_dataloader(
        data_dict['val'], imgsz, batch_size * 2, int(model.stride.max()),
        hyp=hyp, cache=opt.cache, rect=True, rank=-1, workers=opt.workers * 2, pad=0.5, prefix='val: ')[0]

    optimizer = torch.optim.SGD(
        model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'],
        nesterov=True, weight_decay=hyp['weight_decay'])
    
    lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    compute_loss = ComputeLossKD(model, teacher_model, hyp, opt.lambda_kd, device=device)
    ema = ModelEMA(model)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    nb = len(train_loader)
    best_fitness = 0.0
    start_epoch = 0
    nw = max(round(hyp['warmup_epochs'] * nb), 100)

    log_csv_path = save_dir / 'results.csv'
    log_headers = ['epoch', 'gpu_mem', 'box_loss', 'dfl_loss', 'cls_loss', 'kd_box', 'kd_cls', 'kd_obj',
                   'P', 'R', 'mAP_0.5', 'mAP_0.5:0.95', 'val_box_loss', 'val_obj_loss', 'val_cls_loss', 'lr']
    with open(log_csv_path, 'w', newline='') as f: writer = csv.writer(f); writer.writerow(log_headers)

    t0 = time.time()
    LOGGER.info(f'Starting training for {epochs} epochs...')
    
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(6, device=device)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 10) % ('Epoch', 'GPU_mem', 'box_loss', 'dfl_loss', 'cls_loss', 'kd_box', 'kd_cls', 'kd_obj', 'Instances', 'Size'))
        pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)

        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)

            if ni <= nw:
                xi = [0, nw]
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x: x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss, loss_items = compute_loss(preds, targets, imgs, epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema: ema.update(model)

            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_description(('%11s' * 2 + '%11.4g' * 8) %
                                 (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
        
        scheduler.step()
        
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        results, _, _ = validate.run(data_dict, batch_size=batch_size * 2, imgsz=imgsz, model=ema.ema,
                                     single_cls=False, dataloader=val_loader, save_dir=save_dir,
                                     plots=False, callbacks=callbacks, compute_loss=compute_loss)
        
        fi = fitness(np.array(results).reshape(1, -1))
        
        if fi > best_fitness:
            best_fitness = fi
            torch.save({'epoch': epoch, 'model': deepcopy(de_parallel(model)).half(), 'ema': deepcopy(ema.ema).half()}, best)
            LOGGER.info(f"✅ New best model saved to {best} at epoch {epoch} with fitness {fi.item():.4f}")

        torch.save({'epoch': epoch, 'model': deepcopy(de_parallel(model)).half(), 'ema': deepcopy(ema.ema).half()}, last)
        
        current_lr = optimizer.param_groups[0]['lr']
        log_data = [epoch, mem] + list(mloss.cpu().numpy()) + list(results) + [current_lr]
        with open(log_csv_path, 'a', newline='') as f: writer = csv.writer(f); writer.writerow([f"{x:.5g}" if isinstance(x, (float, np.floating)) else x for x in log_data])

    LOGGER.info(f"✅ Training completed in {(time.time() - t0) / 3600:.3f} hours.")
    LOGGER.info(f"Results saved to {save_dir}")
    
    LOGGER.info("Validating best.pt...")
    model = torch.load(best)['ema'].float().to(device)
    results, _, _ = validate.run(data_dict=data_dict, batch_size=batch_size * 2, imgsz=imgsz, model=model,
                                 dataloader=val_loader, save_dir=save_dir, plots=True, callbacks=callbacks)
    
    LOGGER.info(f"Final validation results: P={results[0]:.4f}, R={results[1]:.4f}, mAP@.5={results[2]:.4f}, mAP@.5-.95={results[3]:.4f}")
    
    for f in [last, best]:
        if f.exists(): strip_optimizer(f)
    
    plot_results(file=log_csv_path)

# ======================================
# Command Line Argument Parser
# ======================================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-weights', type=str, required=True, help='Path to teacher model weights (.pt file)')
    parser.add_argument('--student-weights', type=str, default='', help='Path to student model initial weights (optional, for fine-tuning)')
    parser.add_argument('--cfg', type=str, required=True, help='Path to student model configuration file (.yaml)')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset configuration file (.yaml)')
    parser.add_argument('--hyp', type=str, required=True, help='Path to hyperparameters file (.yaml)')
    parser.add_argument('--imgsz', type=int, default=640, help='Train and validation image size (pixels)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Total batch size for all GPUs')
    parser.add_argument('--device', type=str, default='', help='CUDA device, e.g., 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/train_kd', help='Save results to project/name')
    parser.add_argument('--name', default='exp', help='Save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing project/name ok, do not increment')
    parser.add_argument('--workers', type=int, default=8, help='Maximum number of dataloader workers')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='Cache images in RAM or on disk')
    parser.add_argument('--rect', action='store_true', help='Rectangular training')

    # KD-related parameters
    parser.add_argument('--lambda-kd', type=float, default=1.0, help='Weight for the classification distribution KD loss')

    return parser.parse_args()


def main(opt):
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    device = select_device(opt.device, batch_size=opt.batch_size)
    train(opt.hyp, opt, device, Callbacks())


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)