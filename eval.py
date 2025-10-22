import logging
import torch.nn.functional as F
import torchvision.transforms
from torchvision import transforms
from models import ICISPGen
from utils.logger import setup_logger
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
import glob

warnings.filterwarnings("ignore")


# print(torch.cuda.is_available())


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.).item())


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, default='./test_model/rateweight5_checkpoint_best.pth.tar',
                        help="Path to a checkpoint")
    parser.add_argument("--data", type=str, default='/home/ubuntu/Disk/dataset/compress/USTC-TD/',
                        help="Path to dataset")
    parser.add_argument('--dataset_name', type=str, default='3840x2160')
    parser.add_argument('--lmbda', type=float, default=5)
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    save_path = './test_results/{}_{}'.format(str(args.lmbda), args.dataset_name)
    os.makedirs(save_path, exist_ok=True)
    save_img_path = os.path.join(save_path, 'results')
    os.makedirs(save_img_path, exist_ok=True)

    setup_logger('test', save_path, 'test_' + str(args.lmbda), level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    logger_test.info(f"cuda is available: {torch.cuda.is_available()}")

    p = 128
    if args.dataset_name == 'CLIC':
        samples = sorted(glob.glob(os.path.join(args.data, args.dataset_name, 'professional_valid_2020', '*.png')))
    else:
        samples = sorted(glob.glob(os.path.join(args.data, args.dataset_name, '*.png')))

    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = ICISPGen(config=[1, 1, 1, 1, 1, 1], N=64, M=320)
    net = net.to(device)
    net.eval()

    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        logger_test.info(f"Loading: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    if args.real:
        net.update()
        for img_path in samples:
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            img = img.unsqueeze(0)
            img_padded, padding = pad(img, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_enc = net.compress(img_padded)
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)

                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                num_pixels = img.size(0) * img.size(2) * img.size(3)
                logger_test.info(f'Image name: {img_path}')
                logger_test.info(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.4f}bpp')
                logger_test.info(f'MS-SSIM: {compute_msssim(img, out_dec["x_hat"]):.6f}dB')
                logger_test.info(f'PSNR: {compute_psnr(img, out_dec["x_hat"]):.6f}dB')

                # save image
                rec = torchvision.transforms.ToPILImage()(out_dec['x_hat'].squeeze())
                img_name = os.path.basename(img_path)[:-4]
                rec.save(os.path.join(save_img_path, img_name + '.png'))

                Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                PSNR += compute_psnr(img, out_dec["x_hat"])
                MS_SSIM += compute_msssim(img, out_dec["x_hat"])

    else:
        for img_path in samples:
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            img = img.unsqueeze(0)
            img_padded, padding = pad(img, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_net = net.forward(img_padded)

                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)
                logger_test.info(f"Image name: {img_path}")
                logger_test.info(f'PSNR: {compute_psnr(img, out_net["x_hat"]):.6f}dB')
                logger_test.info(f'MS-SSIM: {compute_msssim(img, out_net["x_hat"]):.6f}dB')
                logger_test.info(f'Bit-rate: {compute_bpp(out_net):.6f}bpp')

                # save image
                rec = torchvision.transforms.ToPILImage()(out_net['x_hat'].squeeze())
                img_name = os.path.basename(img_path)[:-4]
                rec.save(os.path.join(save_img_path, img_name + '.png'))

                PSNR += compute_psnr(img, out_net["x_hat"])
                MS_SSIM += compute_msssim(img, out_net["x_hat"])
                Bit_rate += compute_bpp(out_net)

    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count

    logger_test.info(f'average_PSNR: {PSNR:.6f}dB')
    logger_test.info(f'average_MS-SSIM: {MS_SSIM:.6f}')
    logger_test.info(f'average_Bit-rate: {Bit_rate:.6f} bpp')
    logger_test.info(f'average_time: {total_time * 1000:.4f} ms')


if __name__ == "__main__":

    main(sys.argv[1:])
