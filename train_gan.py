import argparse
import logging
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import lmbda

from torch.utils.data import DataLoader

from pytorch_msssim import ms_ssim

from models import ICISPGen, Discriminator
from torch.utils.tensorboard import SummaryWriter
import os
from utils.logger import setup_logger, ImageFolder

from models.lpips import LPIPS
from models.loss import GANLoss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self,):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)

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

def configure_dis_optimizers(net, args):
    return optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.dis_lr)

def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger_train, writer,
        current_iter, criterion_gan, critertion_perceptural, net_dis, optimizer_dis, percep_weight, style_weight, gan_weight, rate_weight, lmbda
):
    model.train()
    device = next(model.parameters()).device

    net_dis.train()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        for p in net_dis.parameters():
            p.requires_grad = False

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        percep_loss, style_loss = critertion_perceptural((out_net['x_hat'].clamp(0,1)-0.5)*2, (d-0.5)*2)

        gan_loss = criterion_gan(out_net['x_hat'], True, is_disc=False).mean()
        total_loss = rate_weight * out_criterion['bpp_loss'] + lmbda*255**2*out_criterion['mse_loss'] + percep_loss.mean() + style_loss.mean() + gan_loss
        total_loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        for p in net_dis.parameters():
            p.requires_grad = True

        optimizer_dis.zero_grad()
        pred_d_real = net_dis(d, out_net['y_hat'].detach(), d)
        pred_d_fake = net_dis(out_net['x_hat'].detach(), out_net['y_hat'].detach(), d)
        dis_real_loss = criterion_gan(pred_d_real, True, is_disc=True).mean()
        dis_fake_loss = criterion_gan(pred_d_fake, False, is_disc=True).mean()
        total_loss_dis = dis_fake_loss + dis_real_loss
        total_loss_dis.backward()
        optimizer_dis.step()

        current_iter += 1

        if current_iter % 200 == 0: # 200
            writer.add_scalar('{}'.format('[train]: bpp loss'), out_criterion['bpp_loss'].item(), current_iter)
            writer.add_scalar('{}'.format('[train]: mse loss'), out_criterion['mse_loss'].item(), current_iter)
            writer.add_scalar('{}'.format('[train]: aux loss'), aux_loss.item(), current_iter)
            writer.add_scalar('{}'.format('[train]: perceptual loss'), (percep_loss/percep_weight).mean().item(), current_iter)
            writer.add_scalar('{}'.format('[train]: style loss'), (style_loss/style_weight).mean().item(), current_iter)
            writer.add_scalar('{}'.format('[train]: gan loss'), (gan_loss/gan_weight).item(), current_iter)
            writer.add_scalar('{}'.format('[train]: total loss'), total_loss.item(), current_iter)
            writer.add_scalar('{}'.format('[train]: lr'), optimizer.param_groups[0]['lr'], current_iter)
            writer.add_scalar('{}'.format('[train]: lr_dis'), optimizer_dis.param_groups[0]['lr'], current_iter)
            writer.add_scalar('{}'.format('[train]: dis_real'), dis_real_loss.item(), current_iter)
            writer.add_scalar('{}'.format('[train]: dis_fake'), dis_fake_loss.item(), current_iter)
            writer.add_scalar('{}'.format('[train]: dis_loss'), total_loss_dis.item(), current_iter)

        if i % 200 == 0: # 200
            logger_train.info(f"Train epoch {epoch}: ["
                                  f"{i * len(d)}/{len(train_dataloader.dataset)}"
                                  f" ({100. * i / len(train_dataloader):.0f}%)]"
                                  f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                                  f'\tBpp loss: {out_criterion["bpp_loss"].item():.6f} |'
                                  f'\tAux loss: {aux_loss.item():.6f} |'
                                  f'\tPerceptual loss: {(percep_loss/percep_weight).mean().item():.6f} |'
                                  f'\tStyle loss: {(style_loss/style_weight).mean().item():.6f} |'
                                  f"\tGan loss: {(gan_loss/gan_weight).item():.6f} |")
            logger_train.info(f"Train epoch {epoch}: ["
                                  f"{i * len(d)}/{len(train_dataloader.dataset)}"
                                  f" ({100. * i / len(train_dataloader):.0f}%)]"
                                  f'\tDis Loss: {total_loss_dis.item():.6f} |'
                                  f'\tReal Dis Loss: {dis_real_loss.item():.6f} |'
                                  f"\tFake Dis Loss: {dis_fake_loss.item():.6f}"
                                  )

    return current_iter


def test_epoch(epoch, test_dataloader, model, criterion, logger_test, critertion_perceptural, percep_weight):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    lpips = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            lpips_loss = (critertion_perceptural((out_net['x_hat'].clamp(0,1)-0.5)*2, (d-0.5)*2)[0]/percep_weight).mean()
            lpips.update(lpips_loss)
            psnr.update(compute_psnr(d,out_net['x_hat']))
            loss.update(out_criterion["bpp_loss"]+lpips_loss)

        logger_test.info(f"Test epoch {epoch}: Average losses:"
                         f"\tLoss: {loss.avg:.6f} |"
                         f"\tPSNR: {psnr.avg:.6f} |"
                         f"\tLPIPS: {lpips.avg:.6f} |"
                         f"\tMSE loss: {mse_loss.avg:.6f} |"
                         f"\tBpp loss: {bpp_loss.avg:.6f} |"
                         f"\tAux loss: {aux_loss.avg:.6f}\n")

    return loss.avg, bpp_loss.avg, psnr.avg, lpips.avg


def save_checkpoint(state, is_best, save_path, filename):
    torch.save(state, save_path + "checkpoint_latest.pth.tar")
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
    parser.add_argument('--dis_lr', default=1e-4, type=float,
                        help='Discriminator Learning rate (default: %(default)s)')
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
        default=0.0004,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument('--percep_weight', type=float, default=5)
    parser.add_argument('--style_weight', type=float, default=2000)
    parser.add_argument('--gan_weight', type=float, default=0.8)
    parser.add_argument('--rate_weight', type=float, default=2.5)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    ) # 16
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
    parser.add_argument("--lr_epochDis", nargs='+', type=int)
    parser.add_argument(
        "--continue_train", action="store_true", default=False
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    save_path = os.path.join(args.save_path, str(args.rate_weight))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "_tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "_tensorboard/")

    setup_logger('train', save_path, 'train_' + str(args.rate_weight), level=logging.INFO, screen=True, tofile=True)
    setup_logger('test', save_path, 'test_' + str(args.rate_weight), level=logging.INFO, screen=True, tofile=True)

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

    net_dis = Discriminator()
    net_dis = net_dis.to(device)

    # test parameters
    num_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger_train.info(f"Total Network parameters:{num_param / 1e6} M")

    num_ga_param = sum(p.numel() for p in net.g_a.parameters() if p.requires_grad)
    logger_train.info(f"analysis transform parameters:{num_ga_param / 1e6} M")

    num_gs_param = sum(p.numel() for p in net.g_s.parameters() if p.requires_grad)
    logger_train.info(f"synthesis transform parameters:{num_gs_param / 1e6} M")

    num_ha_param = sum(p.numel() for p in net.h_a.parameters() if p.requires_grad)
    logger_train.info(f"hyper analysis transform parameters:{num_ha_param / 1e6} M")

    num_dis_param = sum(p.numel() for p in net_dis.parameters() if p.requires_grad)
    logger_train.info(f"discriminator parameters:{num_dis_param / 1e6} M")

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        net_dis = CustomDataParallel(net_dis)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    optimizer_dis = configure_dis_optimizers(net_dis, args)
    milestones = args.lr_epoch
    milestones_dis = args.lr_epochDis
    logger_train.info(f"generator milestones:{milestones}")
    logger_train.info(f"discriminator milestones:{milestones_dis}")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5, last_epoch=-1)
    lr_scheduler_dis = optim.lr_scheduler.MultiStepLR(optimizer_dis, milestones_dis, gamma=0.5, last_epoch=-1)

    criterion = RateDistortionLoss()
    gan_weight = args.gan_weight # 0.8
    criterion_gan = GANLoss(gan_type='hinge', real_label_val=1.0, fake_label_val=0.0, loss_weight=gan_weight).to(device)
    percep_weight = args.percep_weight # 1.0
    style_weight = args.style_weight # 2000
    critertion_perceptural = LPIPS(perceptual_weight=percep_weight,
                 style_weight=style_weight,
                 inp_range=(-1, 1),
                 use_dropout=True,
                 style_measure='L1').to(device)

    last_epoch = 1
    current_iter = 0
    if args.checkpoint:  # load from previous checkpoint
        logger_train.info(f"Loading: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            net_dis.load_state_dict(checkpoint["net_dis_state_dict"])
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_dis.load_state_dict(checkpoint["optimizer_dis"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            lr_scheduler_dis.load_state_dict(checkpoint["lr_scheduler_dis"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs + 1):
        logger_train.info(f"Generator Learning rate: {optimizer.param_groups[0]['lr']}")
        logger_train.info(f"Discriminator Learning rate: {optimizer_dis.param_groups[0]['lr']}")
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
            criterion_gan, critertion_perceptural, net_dis, optimizer_dis, percep_weight, style_weight, gan_weight, args.rate_weight, args.lmbda)

        loss, bpp, psnr, lpips = test_epoch(epoch, test_dataloader, net, criterion, logger_test, critertion_perceptural, percep_weight)
        writer.add_scalar('{}'.format('[test]: loss'), loss, epoch)
        writer.add_scalar('{}'.format('[test]: bpp loss'), bpp, epoch)
        writer.add_scalar('{}'.format('[test]: psnr'), psnr, epoch)
        writer.add_scalar('{}'.format('[test]: lpips'), lpips, epoch)
        lr_scheduler.step()
        lr_scheduler_dis.step()

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
                    'lr_scheduler_dis': lr_scheduler_dis.state_dict(),
                    'net_dis_state_dict': net_dis.state_dict(),
                    'optimizer_dis': optimizer_dis.state_dict(),
                },
                is_best,
                save_path,
                save_path + str(epoch) + "_checkpoint.pth.tar",
            )


if __name__ == "__main__":
    main(sys.argv[1:])
