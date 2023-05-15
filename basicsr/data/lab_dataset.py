import cv2
import random
import time
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import rgb2lab
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.fmix import sample_mask


@DATASET_REGISTRY.register()
class LabDataset(data.Dataset):
    """
    Dataset used for Lab colorizaion
    """

    def __init__(self, opt):
        super(LabDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        meta_info_file = self.opt['meta_info_file']
        assert meta_info_file is not None
        if not isinstance(meta_info_file, list):
            meta_info_file = [meta_info_file]
        self.paths = []
        for meta_info in meta_info_file:
            with open(meta_info, 'r') as fin:
                self.paths.extend([line.strip() for line in fin])

        self.min_ab, self.max_ab = -128, 128
        self.interval_ab = 4
        self.ab_palette = [i for i in range(self.min_ab, self.max_ab + self.interval_ab, self.interval_ab)]
        # print(self.ab_palette)

        self.do_fmix = opt['do_fmix']
        self.fmix_params = {'alpha':1.,'decay_power':3.,'shape':(256,256),'max_soft':0.0,'reformulate':False}
        self.fmix_p = opt['fmix_p']
        self.do_cutmix = opt['do_cutmix']
        self.cutmix_params = {'alpha':1.}
        self.cutmix_p = opt['cutmix_p']


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        gt_size = self.opt['gt_size']
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = cv2.resize(img_gt, (gt_size, gt_size))  # TODO: 直接resize是否是最佳方案？
        
        # -------------------------------- (Optional) CutMix & FMix -------------------------------- #
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > self.fmix_p:
            with torch.no_grad():
                lam, mask = sample_mask(**self.fmix_params)
                
                fmix_index = random.randint(0, self.__len__())
                fmix_img_path = self.paths[fmix_index]
                fmix_img_bytes = self.file_client.get(fmix_img_path, 'gt')
                fmix_img = imfrombytes(fmix_img_bytes, float32=True)
                fmix_img = cv2.resize(fmix_img, (gt_size, gt_size))
                
                mask = mask.transpose(1, 2, 0)  # (1, 256, 256) ->  # (256, 256, 1)
                img_gt = mask * img_gt + (1. - mask) * fmix_img
                img_gt = img_gt.astype(np.float32)

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > self.cutmix_p:
            with torch.no_grad():
                cmix_index = random.randint(0, self.__len__())
                cmix_img_path = self.paths[cmix_index]
                cmix_img_bytes = self.file_client.get(cmix_img_path, 'gt')
                cmix_img = imfrombytes(cmix_img_bytes, float32=True)
                cmix_img = cv2.resize(cmix_img, (gt_size, gt_size))

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(cmix_img.shape[:2], lam)

                img_gt[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]


        # ----------------------------- Get gray lq, to tentor ----------------------------- #
        # convert to gray
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_l, img_ab = rgb2lab(img_gt)

        target_a, target_b = self.ab2int(img_ab)

        # numpy to tensor
        img_l, img_ab = img2tensor([img_l, img_ab], bgr2rgb=False, float32=True)
        target_a, target_b = torch.LongTensor(target_a), torch.LongTensor(target_b)
        return_d = {
            'lq': img_l,
            'gt': img_ab,
            'target_a': target_a,
            'target_b': target_b,
            'lq_path': gt_path,
            'gt_path': gt_path
        }
        return return_d

    def ab2int(self, img_ab):
        img_a, img_b = img_ab[:, :, 0], img_ab[:, :, 1]
        int_a = (img_a - self.min_ab) / self.interval_ab
        int_b = (img_b - self.min_ab) / self.interval_ab

        return np.round(int_a), np.round(int_b)

    def __len__(self):
        return len(self.paths)


def rand_bbox(size, lam):
    '''cutmix 的 bbox 截取函数
    Args:
        size : tuple 图片尺寸 e.g (256,256)
        lam  : float 截取比例
    Returns:
        bbox 的左上角和右下角坐标
        int,int,int,int
    '''
    W = size[0]  # 截取图片的宽度
    H = size[1]  # 截取图片的高度
    cut_rat = np.sqrt(1. - lam)  # 需要截取的 bbox 比例
    cut_w = np.int(W * cut_rat)  # 需要截取的 bbox 宽度
    cut_h = np.int(H * cut_rat)  # 需要截取的 bbox 高度

    cx = np.random.randint(W)  # 均匀分布采样，随机选择截取的 bbox 的中心点 x 坐标
    cy = np.random.randint(H)  # 均匀分布采样，随机选择截取的 bbox 的中心点 y 坐标

    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # 左上角 x 坐标
    bby1 = np.clip(cy - cut_h // 2, 0, H)  # 左上角 y 坐标
    bbx2 = np.clip(cx + cut_w // 2, 0, W)  # 右下角 x 坐标
    bby2 = np.clip(cy + cut_h // 2, 0, H)  # 右下角 y 坐标
    return bbx1, bby1, bbx2, bby2