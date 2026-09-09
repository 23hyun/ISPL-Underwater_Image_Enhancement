import torch
import torch.nn as nn
import argparse
import dataloader_SH_UIEB_AUG_CCLNET
import torch.optim
import torchvision
from torchvision.io import ImageReadMode
import shutil
import model
import os
import numpy as np
import uiqm_plip_pro_0405 as uiqm  # tuple 반환 버전

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def validate_folder_and_log_uiqm(config, model, epoch_idx: int):
    in_dir  = config.val_images_path
    gt_dir  = config.val_clean_images_path
    out_dir = os.path.join(config.val_output_folder, f"val_{epoch_idx}")
    os.makedirs(out_dir, exist_ok=True)

    image_names = sorted(os.listdir(in_dir))
    
    # mode change
    model.eval()
    uiqm_list = []
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for image_name in image_names:
            image_path = os.path.join(in_dir, image_name)
            gt_path = os.path.join(gt_dir, image_name)

            img_haze = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB) \
                        .unsqueeze(0).float() / 255.0
            img_haze = img_haze.cuda(non_blocking=True)

            output = model(img_haze)  # (1,3,H,W)

            saved_path = os.path.join(out_dir, f"{image_name}.png")
            torchvision.utils.save_image(output, saved_path)

            out_img = torchvision.io.read_image(saved_path, mode=ImageReadMode.RGB) \
                      .float().cpu().numpy().transpose(1, 2, 0) / 255.0  # (H,W,3) RGB [0,1]

            out_img = np.clip(out_img, 0, 1)
            out_bgr_255 = (out_img * 255.0)[:, :, ::-1]  # BGR [0,255]

            # uiqm.getUIQM returns: (uiqm_norm, uiqm_raw, uicm, uism, uiconm)
            _, uiqm_raw, _, _, _ = uiqm.getUIQM(out_bgr_255)
            uiqm_list.append(float(uiqm_raw))

            if not os.path.exists(gt_path):
                print(f"[Warning] GT image not found, PSNR/SSIM skip: {gt_path}")
                continue

            img_gt = torchvision.io.read_image(gt_path, mode=ImageReadMode.RGB) \
                     .unsqueeze(0).float() / 255.0
            img_gt = img_gt.cuda(non_blocking=True)

            if img_gt.shape[-2:] != output.shape[-2:]:
                img_gt = torch.nn.functional.interpolate(
                    img_gt,
                    size=output.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            gt_img = img_gt.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            gt_img = np.clip(gt_img, 0, 1)

            psnr_score = psnr(gt_img, out_img, data_range=1.0)
            ssim_score = ssim(gt_img, out_img, channel_axis=2, data_range=1.0)

            psnr_list.append(float(psnr_score))
            ssim_list.append(float(ssim_score))

    mean_uiqm = float(np.mean(uiqm_list)) if uiqm_list else 0.0
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0

    # iter/iter.txt 기록 (epoch_idx: 1부터)
    os.makedirs(config.test_folder, exist_ok=True)
    with open(os.path.join(config.test_folder, "iter.txt"), "a") as f:
        f.write(
            f"epoch:{epoch_idx}  "
            f"uiqm_mean:{mean_uiqm:.4f}  "
            f"psnr_mean:{mean_psnr:.4f}  "
            f"ssim_mean:{mean_ssim:.4f}\n"
        )

    model.train() 
    return mean_uiqm, mean_psnr, mean_ssim


def train(config):
    Transmission_Net = model.Transmission_Net().cuda()
    optimizer_T = torch.optim.Adam(
        Transmission_Net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    criterion = nn.MSELoss().cuda()

    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    LOss_T = np.zeros((config.num_epochs, len(train_loader)), dtype=np.float32)

    # checkpoint가 있다면 여기 값을 바꾸면 됨
    checkpoint = 0

    for epoch in range(config.num_epochs):
        Transmission_Net.train()

        epoch_idx = epoch + checkpoint + 1

        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig = img_orig.cuda(non_blocking=True)
            img_haze = img_haze.cuda(non_blocking=True)

            restored_image = Transmission_Net(img_haze)
            loss = criterion(restored_image, img_orig)

            optimizer_T.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Transmission_Net.parameters(), config.grad_clip_norm)
            optimizer_T.step()

            LOss_T[epoch, iteration] = float(loss.item())

            if ((iteration + 1) % config.display_iter) == 0:
                print(f"Epoch {epoch_idx}, Iter {iteration+1} | Loss: {loss.item():.4f}")

            # sample 저장
            sample_dir = os.path.join(config.sample_output_folder, f"Sample_{epoch_idx}")
            os.makedirs(sample_dir, exist_ok=True)

            if (iteration + 1) % 10 == 1:
                torchvision.utils.save_image(
                    torch.cat((img_haze, img_orig, restored_image), 0),
                    os.path.join(sample_dir, f"{iteration + 1}.jpg")
                )

            # snapshot 저장 
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(
                    Transmission_Net.state_dict(),
                    os.path.join(config.snapshots_folder, f"Epoch_optimize_confirm{epoch_idx}.pth")
                )

        # epoch 끝날 때 snapshot 1회 저장 
        torch.save(
            Transmission_Net.state_dict(),
            os.path.join(config.snapshots_folder, f"Epoch_optimize_confirm{epoch_idx}.pth")
        )

        # epoch 평균 loss 기록 
        train_loss_mean = float(np.mean(LOss_T[epoch, :])) if len(train_loader) > 0 else 0.0
        os.makedirs(config.test_folder, exist_ok=True)
        with open(os.path.join(config.test_folder, "loss.txt"), "a") as f:
            f.write(f"epoch:{epoch_idx}  loss_mean:{train_loss_mean:.6f}\n")

        # validation: val_1, val_2 ... /
        mean_uiqm, mean_psnr, mean_ssim = validate_folder_and_log_uiqm(
            config,
            Transmission_Net,
            epoch_idx
        )

        print(
            f"Epoch [{epoch_idx}/{config.num_epochs}] - "
            f"Validation | "
            f"UIQM: {mean_uiqm:.4f}, "
            f"PSNR: {mean_psnr:.4f}, "
            f"SSIM: {mean_ssim:.4f}"
        )

        # 최종 모델 저장(매 epoch 덮어쓰기)
        torch.save(
            Transmission_Net.state_dict(),
            os.path.join(config.snapshots_folder, "dehazer_optimize_final_confirm.pth")
        )


if __name__ == "__main__":
    dataset = "train"

    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_images_path', type=str, default="data\\" + dataset + "B\\") # train clean image
    parser.add_argument('--hazy_images_path', type=str, default="data\\" + dataset + "A\\") # train image
    parser.add_argument('--val_images_path', type=str, default="validation image path here\\")
    parser.add_argument('--val_clean_images_path', type=str, default="validation clean image path here\\")

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.00025)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=20)

    parser.add_argument('--snapshots_folder', type=str, default="snapshots\\")
    parser.add_argument('--sample_output_folder', type=str, default="samples\\")
    parser.add_argument('--val_output_folder', type=str, default="val\\")
    parser.add_argument('--test_folder', type=str, default="iter\\")

    config = parser.parse_args()

    if os.path.exists(config.snapshots_folder):
        shutil.rmtree(config.snapshots_folder)
    if os.path.exists(config.sample_output_folder):
        shutil.rmtree(config.sample_output_folder)
    if os.path.exists(config.test_folder):
        shutil.rmtree(config.test_folder)

    os.makedirs(config.val_output_folder, exist_ok=True)
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.sample_output_folder, exist_ok=True)
    os.makedirs(config.test_folder, exist_ok=True)

    train(config)