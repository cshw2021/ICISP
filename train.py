import argparse
import logging
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim

from models import ICISPGen
from torch.utils.tensorboard import SummaryWriter
import os
from utils.logger import setup_logger, ImageFolder

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = [
        p
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles")
    ]
    aux_parameters = [
        p
        for n, p in net.named_parameters()
        if n.endswith(".quantiles")
    ]

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = set(parameters) & set(aux_parameters)
    union_params = set(parameters) | set(aux_parameters)

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (p for p in parameters if p.requires_grad),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (p for p in aux_parameters if p.requires_grad),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, writer,
        current_iter, type='mse'
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        current_iter += 1

        if current_iter % 200 == 0:
            writer.add_scalar('{}'.format('[train]: total loss'), out_criterion['loss'].item(), current_iter)
            writer.add_scalar('{}'.format('[train]: bpp loss'), out_criterion['bpp_loss'].item(), current_iter)
            writer.add_scalar('{}'.format('[train]: aux loss'), aux_loss.item(), current_iter)
            writer.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_iter)

        if i % 200 == 0: #
            if type == 'mse':
                logger_train.info(f"Train epoch {epoch}: ["
                                  f"{i * len(d)}/{len(train_dataloader.dataset)}"
                                  f" ({100. * i / len(train_dataloader):.0f}%)]"
                                  f'\tLoss: {out_criterion["loss"].item():.6f} |'
                                  f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                                  f'\tBpp loss: {out_criterion["bpp_loss"].item():.6f} |'
                                  f"\tAux loss: {aux_loss.item():.6f}")
            else:
                logger_train.info(f"Train epoch {epoch}: ["
                                  f"{i * len(d)}/{len(train_dataloader.dataset)}"
                                  f" ({100. * i / len(train_dataloader):.0f}%)]"
                                  f'\tLoss: {out_criterion["loss"].item():.6f} |'
                                  f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.6f} |'
                                  f'\tBpp loss: {out_criterion["bpp_loss"].item():.6f} |'
                                  f"\tAux loss: {aux_loss.item():.6f}")

    return current_iter


def test_epoch(epoch, test_dataloader, model, criterion, logger_test, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()
        psnr = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                psnr.update(compute_psnr(d,out_net['x_hat']))

        logger_test.info(f"Test epoch {epoch}: Average losses:"
                         f"\tLoss: {loss.avg:.6f} |"
                         f"\tPSNR: {psnr.avg:.6f} |"
                         f"\tMSE loss: {mse_loss.avg:.6f} |"
                         f"\tBpp loss: {bpp_loss.avg:.6f} |"
                         f"\tAux loss: {aux_loss.avg:.6f}\n")
    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        logger_test.info(f"Test epoch {epoch}: Average losses:"
                         f"\tLoss: {loss.avg:.3f} |"
                         f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
                         f"\tBpp loss: {bpp_loss.avg:.2f} |"
                         f"\tAux loss: {aux_loss.avg:.2f}\n")

    return loss.avg, bpp_loss.avg, psnr.avg


def save_checkpoint(state, is_best, epoch, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, save_path + "checkpoint_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/home/ubuntu/Disk/dataset/compress/', help="Training dataset"
    )
    parser.add_argument('--dataset_name', type=str, default='LSDIR')
    parser.add_argument(
        "-vd", "--val_dataset", type=str, default='/home/ubuntu/Disk/dataset/compress/', help="Validation dataset"
    )
    parser.add_argument('--val_dataset_name', type=str, default='Kodak')
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=256,
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "_tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "_tensorboard/")

    setup_logger('train', save_path, 'train_' + str(args.lmbda), level=logging.INFO, screen=True, tofile=True)
    setup_logger('test', save_path, 'test_' + str(args.lmbda), level=logging.INFO, screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_test = logging.getLogger('test')

    train_dataset = ImageFolder(args.dataset, args.dataset_name, args.patch_size, split="train")
    test_dataset = ImageFolder(args.val_dataset, args.val_dataset_name, 512, split="test")

    logger_train.info(args)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logger_train.info(device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = ICISPGen(config=[1, 1, 1, 1, 1, 1], N=args.N, M=320)
    net = net.to(device)

    # test parameters
    num_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger_train.info(f"Total Network parameters:{num_param / 1e6} M")

    num_ga_param = sum(p.numel() for p in net.g_a.parameters() if p.requires_grad)
    logger_train.info(f"analysis transform parameters:{num_ga_param / 1e6} M")

    num_gs_param = sum(p.numel() for p in net.g_s.parameters() if p.requires_grad)
    logger_train.info(f"synthesis transform parameters:{num_gs_param / 1e6} M")

    num_ha_param = sum(p.numel() for p in net.h_a.parameters() if p.requires_grad)
    logger_train.info(f"hyper analysis transform parameters:{num_ha_param / 1e6} M")

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = args.lr_epoch
    logger_train.info(f"milestones:{milestones}")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5, last_epoch=-1)

    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 1
    current_iter = 0
    if args.checkpoint:  # load from previous checkpoint
        logger_train.info(f"Loading: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs + 1):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_iter = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            writer,
            current_iter,
            type
        )
        loss, bpp, psnr = test_epoch(epoch, test_dataloader, net, criterion, logger_test, type)
        writer.add_scalar('{}'.format('[test]: loss'), loss, epoch)
        writer.add_scalar('{}'.format('[test]: bpp loss'), bpp, epoch)
        writer.add_scalar('{}'.format('[test]: psnr'), psnr, epoch)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
