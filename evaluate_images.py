import glob
import imageio
import numpy as np
import torch.nn as nn
import torch
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
import piq
import logging

device = torch.device("cuda:0")
logging.basicConfig(filename='Urban100_results.log', level=logging.INFO)

path1 = "/home/emrecan/Desktop/Comp411/data/results/Urban100/"
path2 = "/home/emrecan/Desktop/Comp411/data/HR/Urban100/"

logging.info("Predicted Image path: " + str(path1))
logging.info("HR Image path: " + str(path2))
logging.info("-----------------------------------\n")


class SRFlowDataset(Dataset):
    def __init__(self, img_path):
        self.imgs_path1 = img_path
        file_list = glob.glob(self.imgs_path1 + "*")
        # print(file_list)
        self.data = []
        for img_path1 in natsorted(glob.glob(self.imgs_path1 + "/*.png")):
            self.data.append(img_path1)
        #print(self.data)
        # print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path1 = self.data[idx]
        img1 = imageio.imread(img_path1)
        img1 = (img1)/255.0
        logging.info("Image Name:" + str(img_path1))
        logging.info("Image Shape:" + str(img1.shape))
        # print(img_path1)
        # print(img1.shape)
        h, w, c = img1.shape
        h = (h//4)*4
        w = (w//4)*4
        img_tensor1 = torch.from_numpy(img1[:h, :w, :])
        img_tensor1 = img_tensor1.permute(2, 0, 1)
        img_tensor1 = img_tensor1.to(device).float()
        return {'images': img_tensor1, 'labels': idx}


dataset_01, dataset_div2k = SRFlowDataset(path1), SRFlowDataset(path2)
data_loader01 = DataLoader(dataset_01, batch_size=1, shuffle=False)
data_loaderdiv2k = DataLoader(dataset_div2k, batch_size=1, shuffle=False)

lpips_vgg = piq.LPIPS()
pieapp_metric = piq.PieAPP()
dists_metric = piq.DISTS()
fid_metric = piq.FID()
is_metric = piq.IS()
loss_mse = nn.MSELoss()

first_feats = fid_metric.compute_feats(data_loader01)
second_feats = fid_metric.compute_feats(data_loaderdiv2k)
test_fid = fid_metric(first_feats, second_feats)


first_feats_is = is_metric.compute_feats(data_loader01)
second_feats_is = is_metric.compute_feats(data_loaderdiv2k)
test_is = is_metric(first_feats, second_feats)


test_psnr = 0
test_ssim = 0
test_msssim = 0
test_lpips = 0
test_pieapp = 0
test_dists = 0
test_mse = 0

num_of_images = len(data_loader01)

for data01, datahr in zip(data_loader01, data_loaderdiv2k):
    # print(data['images'].shape)
    # print(data['labels'])
    img_pred = data01['images']
    img_hr = datahr['images']

    psnr_img = piq.psnr(img_pred, img_hr)
    test_psnr += psnr_img
    logging.info("PSNR:" + str(psnr_img))
    
    ssim_img = piq.ssim(img_pred, img_hr)
    test_ssim += ssim_img
    logging.info("SSIM:" + str(ssim_img))

    msssim_img = piq.multi_scale_ssim(img_pred, img_hr)
    test_msssim += msssim_img
    logging.info("MS SSIM:" + str(msssim_img))

    lpips_img = lpips_vgg(img_pred, img_hr)
    test_lpips += lpips_img
    logging.info("LPIPS VGG:" + str(lpips_img))

    pieapp_img = pieapp_metric(img_pred, img_hr)
    test_pieapp += pieapp_img
    logging.info("PieAPP:" + str(pieapp_img))

    dists_img = dists_metric(img_pred, img_hr)
    test_dists += dists_img
    logging.info("DISTS:" + str(dists_img))

    mse_img = loss_mse(img_pred, img_hr)
    test_mse += mse_img
    logging.info("MSE:" + str(mse_img))

    logging.info("-----------------------------------\n")

test_psnr /= num_of_images
test_ssim /= num_of_images
test_msssim /= num_of_images
test_lpips /= num_of_images
test_pieapp /= num_of_images
test_dists /= num_of_images
test_mse /= num_of_images

logging.info("-----------------------------------\n")
logging.info("PSNR:" + str(test_psnr))
logging.info("SSIM:" + str(test_ssim))
logging.info("MS-SSIM: " + str(test_msssim))
logging.info("LPIPS VGG:" + str(test_lpips))
logging.info("FID Score:" + str(test_fid))
logging.info("IS:" + str(test_is))
logging.info("PieAPP:" + str(test_pieapp))
logging.info("DISTS:" + str(test_dists))
logging.info("MSE:" + str(test_mse))



