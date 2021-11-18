import argparse
import os
from tqdm import tqdm

import apex
from apex import amp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast
import torch.optim as optim

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score,recall_score
import numpy as np
import pandas as pd

from src.solver import make_lr_scheduler, make_optimizer
from src.data import Dataset_Custom_3d
from src.modeling import WeightedBCEWithLogitsLoss
from src.modeling import ConvLSTM3D, ResNet3D, Cnn_ConvLSTM
from src.config import get_cfg, get_cfg_cnn_lstm
from src.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
            help="config yaml path")
    parser.add_argument("--load", type=str, default="",
            help="path to model weight")
    parser.add_argument("-ft", "--finetune", action="store_true",
        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
        help="model running mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
        help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
        help="enable evaluation mode for testset")
    parser.add_argument("-d", "--debug", action="store_true",
        help="enable debug mode for test")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args
    
def build_model(cfg):
    model = ConvLSTM3D(cfg)
    return model

def test_model(_print, cfg, model, test_loader):
    model.eval()

    ids = []
    probs = []
    tbar = tqdm(test_loader)

    with torch.no_grad():
        for i, (image, id_code) in enumerate(tbar):
            image = image.cuda()
            id_code = list(*zip(*id_code))
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            with autocast():
                output = model(image, seq_len)
            output = torch.softmax(output)
            probs.append(output.cpu().numpy())
            ids += id_code

    probs = np.concatenate(probs, 0)
    submit = pd.concat([pd.Series(ids), pd.DataFrame(probs)], axis=1)
    submit.columns = ["image", "abnormal", "normal"]
    return submit
def acc(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct)/y_true.shape[0]
    
def valid_model(_print, cfg, model, valid_loader, valid_criterion):
    
    model.eval()

    preds = []
    targets = []
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            target = target.view(-1, target.size(-1))
            with autocast():
                output = model(image, seq_len)
            preds.append(output.cpu())
            targets.append(target.cpu())

    preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)
    # record loss
    loss_tensor = valid_criterion(preds, targets)
    val_loss = loss_tensor.sum() / valid_criterion.class_weights.sum()
    abnormal_loss = loss_tensor[0]
    normal_loss = loss_tensor[1]
    
    _print("Val. loss: %.5f - abnormal: %.3f - normal: %.3f" % (
        val_loss, abnormal_loss, normal_loss))
    # record AUC
    auc = roc_auc_score(targets[:, :].numpy(), preds[:, :].numpy(), average=None)
    _print("Val. AUC - abnormal: %.3f - normal: %.3f" % (
            auc[0], auc[1]))
    id_preds = []
    id_targets = []
    id_preds = np.argmax(preds.cpu().detach().numpy(), axis= 1)
    id_targets = np.argmax(targets.cpu().detach().numpy(), axis= 1)
    # _print("Val. ACC: %.7f"  % (acc(id_preds,id_targets)))
    _print("Val. ACC: %.7f Precision %.7f Recall %.7f"  % (acc(id_preds,id_targets), precision_score(id_preds,id_targets,average='binary'),
        recall_score(id_preds,id_targets,average='binary')))
    
    return val_loss

def train_loop(_print, cfg, model, train_loader, criterion, valid_loader, valid_criterion, optimizer, scheduler, start_epoch, best_metric):
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"\nEpoch {epoch + 1}")

        losses = AverageMeter()
        acces = AverageMeter()
        model.train()
        tbar = tqdm(train_loader)

        for i, (image, target) in enumerate(tbar):
            image = image.cuda().half()
            target = target.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            target = target.view(-1, target.size(-1))
            
            if cfg.DATA.MIXUP:
                mixed_image, target, mixed_target, lamb = mixup_data(image, target, cfg.DATA.CM_ALPHA)
                
            with autocast():
                output = model(image, seq_len)
                
            if cfg.DATA.MIXUP:
                loss = mixup_criterion(criterion, output, target, mixed_target, lamb)
            else:
                loss = criterion(output, target)
            # gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS

            if cfg.SYSTEM.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            id_preds = np.argmax(output.cpu().detach().numpy(), axis= 1)
            id_targets = np.argmax(target.cpu().detach().numpy(), axis= 1)
            acces.update(acc(id_preds,id_targets),1)
            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            tbar.set_description("Train loss: %.5f, acc: %.5f, learning rate: %.6f" % (losses.avg, acces.avg, optimizer.param_groups[-1]['lr']))
    
        _print("Train loss: %.5f, acc: %.5f, learning rate: %.6f" % (losses.avg, acces.avg, optimizer.param_groups[-1]['lr']))

        loss = valid_model(_print, cfg, model, valid_loader, valid_criterion)
        is_best = loss < best_metric
        best_metric = min(loss, best_metric)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": cfg.EXP,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")

def main(args, cfg):
    logging = setup_logger(args.mode, cfg.DIRS.LOGS, 0, filename=f"{cfg.EXP}.txt")
    logging.info(f" Config: \n{cfg}")
    # Declare variables
    start_epoch = 0
    best_metric = 10.

    # Create model
    model = build_model(cfg)
    # Define Loss and Optimizer
    train_criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(cfg.LOSS.WEIGHTS))
    valid_criterion = WeightedBCEWithLogitsLoss(class_weights=torch.tensor(cfg.LOSS.WEIGHTS), reduction='none')
    optimizer = make_optimizer(cfg, model)
    #train_criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        train_criterion = train_criterion.cuda()

    if cfg.SYSTEM.FP16:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer,
                                          opt_level=cfg.SYSTEM.OPT_L,
                                          keep_batchnorm_fp32=(True if cfg.SYSTEM.OPT_L == "O2" else None))

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            logging.info(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            if not args.finetune:
                logging.info("resuming optimizer ...")
                optimizer.load_state_dict(ckpt.pop('optimizer'))
                start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            logging.info(f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.load}'")
    DataSet = Dataset_Custom_3d
    train_ds = DataSet(cfg, mode="train")
    valid_ds = DataSet(cfg, mode="valid")
    test_ds = DataSet(cfg, mode="test")
    if cfg.DEBUG:
        train_ds = Subset(train_ds, np.random.choice(np.arange(len(train_ds)), 50))
        valid_ds = Subset(valid_ds, np.random.choice(np.arange(len(valid_ds)), 20))

    train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=True,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, 1,
                            pin_memory=False, shuffle=False,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    test_loader = DataLoader(test_ds, 1, pin_memory=False, shuffle=False,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

    scheduler = make_lr_scheduler(cfg, optimizer, train_loader)
    if args.mode == "train":
        train_loop(logging.info, cfg, model, \
                train_loader, train_criterion, valid_loader, valid_criterion, \
                optimizer, scheduler, start_epoch, best_metric)
    elif args.mode == "valid":
        valid_model(logging.info, cfg, model, valid_loader, valid_criterion)
    else:
        submission = test_model(logging.info, cfg, model, test_loader)
        sub_fpath = os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}.csv")
        submission.to_csv(sub_fpath, index=False)
if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
    cfg.freeze()
    # make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])
    # seed, run
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)