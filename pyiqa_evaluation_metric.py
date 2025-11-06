## writen by Hao 2025-11-06
import torch
import torch.cuda
from PIL import Image
from torchvision import transforms
import glob
import os
import logging
from datetime import datetime

import pyiqa
from neuralcompression.metrics import update_patch_fid
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

if __name__ == '__main__':

    # the name of GT and reconstructed images should be the same except for the bpp info appended to the reconstructed images
    # Ground-truth: img01.png
    # Reconstructed: img01_0.123456.png (0.123456 is the bpp info)
    
    rate_weight = '' # give the rate weight info of your reconstructed images, e.g., '1/1.5/2.5/5'
    result_path = '' # give the path of your reconstructed images
    GT_path = ''     # give the path of your ground-truth images

    save_log_path = './test_results/' # save the log file
    os.makedirs(save_log_path, exist_ok=True)

    setup_logger('test', save_log_path, rate_weight, level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    psnr_metric = pyiqa.create_metric('psnr', device=device)
    msssim_metric = pyiqa.create_metric('ms_ssim', device=device)
    dists_metric = pyiqa.create_metric('dists', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    lpips_vgg_metric = pyiqa.create_metric('lpips-vgg', device=device)
    cal_fid = FrechetInceptionDistance().to(device)
    cal_kid = KernelInceptionDistance().to(device)

    count = 0
    total_psnr, total_msssim, total_dists, total_lpips, total_bpp, total_lpips_vgg = 0, 0, 0, 0, 0, 0

    for img_path in sorted(glob.glob(os.path.join(GT_path, '*.png'))):
        count += 1
        img_name = os.path.basename(img_path)[:-4]
        GT_img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device) # [0,1]
        img = transforms.ToTensor()(Image.open(glob.glob(os.path.join(result_path, f"{img_name}_*.png"))[0]).convert('RGB')).to(device) # [0,1]
        bpp = float(os.path.basename(glob.glob(os.path.join(result_path, f"{img_name}_*.png"))[0]).split('_')[-1][:-4])
        img = img.unsqueeze(0)
        GT_img = GT_img.unsqueeze(0)

        psnr_value = psnr_metric(img, GT_img).item()
        ms_ssim_value = msssim_metric(img, GT_img).item()
        dists_value = dists_metric(img, GT_img).item()
        lpips_value = lpips_metric(img, GT_img).item()
        lpips_vgg_value = lpips_vgg_metric(img, GT_img).item()

        with torch.no_grad():
            update_patch_fid(GT_img, img, fid_metric=cal_fid, kid_metric=cal_kid, patch_size=256)

        logger_test.info('==== {} ===='.format(img_path))
        logger_test.info('==== PSNR:{:.6f}\tMS_SSIM:{:.6f}\tBPP:{:.6f}\tLPIPS:{:.6f}\tDISTS:{:.6f}\tLPIPS-VGG:{:.6f} ===='.
                        format(psnr_value, ms_ssim_value, bpp, lpips_value, dists_value, lpips_vgg_value))

        total_psnr += psnr_value
        total_msssim += ms_ssim_value
        total_lpips += lpips_value
        total_dists += dists_value
        total_bpp += bpp


    avg_psnr, avg_ms_ssim, avg_bpp = total_psnr/count, total_msssim/count, total_bpp/count
    avg_lpips, avg_dists = total_lpips/count, total_dists/count
    avg_lpips_vgg = total_lpips_vgg/count

    fid_total = float(cal_fid.compute())
    kid_total = float(cal_kid.compute()[0])

    logger_test.info(
        f'==== {count} images => PSNR: {avg_psnr:.6f}\tMS_SSIM: {avg_ms_ssim:.6f}\tBPP: {avg_bpp:.6f}\t====')
    logger_test.info(
        f'==== LPIPS: {avg_lpips:.6f}\tLPIPS-VGG: {avg_lpips_vgg:.6f}\tDISTS: {avg_dists:.6f}\t====')
    logger_test.info(
        f'==== FID: {fid_total:.6f}\KID: {kid_total:.6f}\t====')












