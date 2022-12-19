import albumentations as A
import albumentations.augmentations.functional as albuF
from albumentations.pytorch import ToTensorV2
import math
from PIL import Image
import numpy as np
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import torch
from time import time
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
global scaler
import torchvision.transforms as T
import csv
import os
from pathlib import Path
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
from utils import utils_image as util
def img_saver(model,dataset,current_step,opt):


    # prepare input and forward
    test_data = dataset[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_name_ext = os.path.basename(test_data['L_path'][0])
    img_name, ext = os.path.splitext(image_name_ext)
    img_dir = os.path.join(opt['path']['images'], img_name)
    util.mkdir(img_dir)
    ## B,C,H,W,
    E_imgs, H_imgs = [], []
    for i, (L, H) in enumerate(zip(test_data['L'], test_data['H'])):  ## change for gpu mem

        model.feed_data({'L': L.unsqueeze(0), 'H': H.unsqueeze(0)})
        model.test()

        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])
        E_imgs.append(E_img)
        H_imgs.append(H_img)

    E_img,H_img= np.stack(E_imgs, axis=0),np.stack(H_imgs, axis=0)
    if E_img.ndim == 3:
        E_img = rearrange(E_img, ' (b h1 w1) h w  ->  (h1 h) (w1 w) b', h1=dataset.H_windows,
                          w1=dataset.W_windows)
        H_img = rearrange(H_img, ' (b h1 w1) h w  ->  (h1 h) (w1 w) b', h1=dataset.H_windows,
                          w1=dataset.W_windows)
    elif E_img.ndim == 4:
        E_img = rearrange(E_img, ' (b h1 w1) h w c -> b (h1 h) (w1 w) c', h1=dataset.H_windows,
                          w1=dataset.W_windows)
        H_img = rearrange(H_img, ' (b h1 w1) h w c -> b (h1 h) (w1 w) c', h1=dataset.H_windows,
                          w1=dataset.W_windows)

    sns.set_style(style="darkgrid")
    plt.figure(figsize=(8, 6))
    # Image.fromarray(E_img[0,:,:,2]).show(title="memax_H_img__")
    # Image.fromarray(H_img[0,:,:,2]).show(title="memax_H_img__")
    sns.distplot(E_img[0,:,:,2].flatten(),color = "red"  , label = "Pred_img")
    sns.distplot(H_img[0,:,:,2].flatten(),color = "green", label = "Good_img")
    plt.legend(title="dist bet good,pred")


    res_save = np.concatenate((E_img, H_img), 1)[:,:,:,2]  ## todo concat dim 3 --> 2 (horizontal cat to vertical)
    save_img_path = os.path.join(os.path.abspath(img_dir), '{:s}_{:d}.png'.format(img_name, current_step))
    save_fig_path = make_dir_tree_from_file(os.path.join(os.path.abspath(img_dir+'/dist_fig/'), '{:s}_{:d}.png'.format('dist_', current_step)))
    plt.savefig(save_fig_path)
    plt.show()
    util.imsave(res_save, save_img_path)
    print(
        "img save done"
    )

def add_noise(img,sigma= [0, 50],sigma_test = 0):
    sigma_min, sigma_max = sigma[0], sigma[1]
    # sigma_test = opt['sigma_test'] if opt['sigma_test'] else 0
    noise_level = torch.FloatTensor([np.random.uniform(sigma_min, sigma_max)]) / 255.0

    noise_level_map = torch.ones((1, img.size(1), img.size(2))).mul_(noise_level).float()
    # torch.full((1, img_L.size(1), img_L.size(2)), noise_level)

    # ---------------------------------
    # add noise
    # ---------------------------------
    noise = torch.randn(img.size()).mul_(noise_level).float()
    img.add_(noise)
    return img

def get_concat_v_pil_3(img: tuple, direction="vertical"):
    from PIL import Image
    total_height = []
    total_width = []
    for i in img:
        total_height.append(i.height)
        total_width.append(i.width)
    if direction == "vertical":
        dst = Image.new('RGB', (total_width[0], sum(total_height)))
        for idx, j in enumerate(img):
            if idx == 0:
                heigt_ = 0
            else:
                heigt_ = total_height[idx - 1]
            dst.paste(j, (0, heigt_))
    elif direction == "horizontal":
        dst = Image.new('RGB', (sum(total_width), total_height[0]))
        for idx, k in enumerate(img):
            if idx == 0:
                widt_ = 0
            else:
                widt_ = total_width[idx - 1]
            dst.paste(k, (widt_, 0))
    # elif str(direction) =="list":
    #     dst = Image.new('RGB', (sum(total_width),total_height[0]))
    #     for idx,k in enumerate(img):
    #         if idx == 0: widt_ = 0
    #         else : widt_ = total_width[idx-1]
    #         dst.paste(j, (widt_, 0))
    else:
        raise Exception("뭬에에에엥 !!")

    return dst


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def minmax(x,max = 1):
    xmin = np.min(x)
    xmax = np.max(x)
    f = lambda x: (x - xmin / (xmax - xmin))*max
    mapped_x = f(x)
    print(np.min(mapped_x))
    print(np.max(mapped_x))
    return mapped_x



def make_batch(x, window_size,final_row = False):
    from einops import rearrange, reduce, repeat
    windows_w, windows_h = x.shape[-1] / window_size, x.shape[-2] / window_size
    if str(type(x)) == "<class 'numpy.ndarray'>":
        if x.ndim == 2:
            x=np.expand_dims(x,0)
            output = rearrange(x, 'c (h1 h) (w1 w)  ->(h1 w1) c h w ', h1=int(windows_h), w1=int(windows_w))
        elif x.ndim == 3:
            output = rearrange(x, 'c (h1 h) (w1 w)  ->(h1 w1) c h w ', h1=int(windows_h), w1=int(windows_w))
    if str(type(x)) == "<class 'torch.Tensor'>":
        if x.ndim == 2:
            x=x.unsqueeze(0)
            output = rearrange(x, 'c (h1 h) (w1 w)  ->(h1 w1) c h w ', h1=int(windows_h), w1=int(windows_w))
        elif x.ndim == 3:
            output = rearrange(x, 'c (h1 h) (w1 w)  ->(h1 w1) c h w ', h1=int(windows_h), w1=int(windows_w))
    if final_row == True:
        output = output[::windows_h]
    return output


def padding_3chan(x, window_size=224, crop=False,outputchan=3,gradation=True):
    # if str(type(x)) == "<class 'numpy.ndarray'>":
    #     if len(np.shape(x)) < 3:
    #         x = np.expand_dims(x,0)
    # if str(type(x)) == "<class 'torch.Tensor'>":
    #     if len(x.shape) < 3:
    #         x = x.unsqueeze(0)
    x_H, x_W = x.shape[-2], x.shape[-1]
    resize_H, resize_W = (round_to_multiple(x_H, window_size) - x_H) // 2, (
                round_to_multiple(x_W, window_size) - x_W) // 2
    if resize_W < 0 or resize_H < 0:
        if resize_W < 0:
            x = x[:, -resize_W:resize_W]
            resize_W = 0
        if resize_H < 0:
            x = x[-resize_H:resize_H, :]
            resize_H = 0
    p1d = ((resize_H, resize_H),(resize_W, resize_W))
    x = np.pad(x, p1d, 'constant', constant_values=0)
    x=sklearn_minmax(x)
    if gradation==True:
        sigmoid_contrast()
    else :
        out = repeat(x, ' h w ->c h w', c=outputchan)

    # if str(type(x)) == "<class 'numpy.ndarray'>":
    #     x = torch.tensor(x)
    # out = F.pad(, p1d, "constant", 0)
    # if str(type(x)) == "<class 'numpy.ndarray'>":
    #     x = torch.tensor(x)
    #     out = F.pad(torch.concat((x.clone().detach(), x.clone().detach(), x.clone().detach()), 0), p1d, "constant", 0)
    # else:
    #     out = F.pad(torch.concat((x.clone().detach(), x.clone().detach(), x.clone().detach()), 0), p1d, "constant", 0)
    if crop:
        out = img_crop_and_make_batch(out, window_size)
    return out


def pad_for_window(x, window_size=224):
    x_H, x_W = x.shape[-2], x.shape[-1]
    resize_H, resize_W = (round_to_multiple(x_H, window_size) - x_H) // 2, (
            round_to_multiple(x_W, window_size) - x_W) // 2
    if resize_W < 0 or resize_H < 0:
        if resize_W < 0:
            x = x[:, -resize_W:resize_W]
            resize_W = 0
        if resize_H < 0:
            x = x[-resize_H:resize_H, :]
            resize_H = 0
    p1d = ((resize_H, resize_H), (resize_W, resize_W))
    x = np.pad(x, p1d, 'constant', constant_values=0)
    return x


from tqdm.auto import tqdm
import numpy as np
import torchvision.transforms as T
import torch
import csv
import os
def compute_mean_std(img_list,mean_file = 'mean_std.csv',force_refresh=True):
    transform_ = T.Compose(
        [T.ToTensor()])  # 0~1 transform#normalize
    # f = open('file_crop_size.csv', 'w', newline='')
    # wr = csv.writer(f)
    # wr.writerow(["path","x_min",'x_max','x_size','y_min','y_max','y_size'])
    # f.close()
    if os.path.isfile(mean_file) and force_refresh == False:
        csv_data = np.loadtxt(mean_file,delimiter=",")
        _mean=csv_data[0,:]
        _std =csv_data[1,:]
        _min =csv_data[2,:]
        _max =csv_data[3,:]
        print('==> got mean and std from {}..'.format(mean_file))
    else:
        print('==> Computing mean and std..')
        before = time()

        ## just know image chanel
        if img_list[0][-4:] == '.npy':
            image = transform_(np.load(img_list[0])).permute((1, 2, 0)).contiguous()  ## current numpy is CHW but transform will turn to HWC
        else:
            image = transform_(Image.open(img_list[0]).convert('RGB'))

        _mean = torch.zeros(image.size()[0])
        _std = torch.zeros(image.size()[0])
        _min = torch.zeros(image.size()[0])
        _max = torch.zeros(image.size()[0])
        # loop through images
        for inputs in  tqdm(img_list):
            if inputs[-4:] == '.npy':
                image = torch.tensor(np.load(inputs))
            else:
                image = torch.tensor(Image.open(inputs).convert('RGB'))
            _mean = torch.add(_mean, torch.mean(image, dim=[1, 2]))
            _std = torch.add (_std, torch.std(image, dim=[1, 2]))
            _min = torch.add(_min , image.view(image.shape[0], -1).min(1).values)
            _max = torch.add(_max , image.view(image.shape[0], -1).max(1).values)
            # a.view(a.shape[0], -1).min(1).values
            # _q3  = np.percentile(arr, 75)
            # _q1  = np.percentile(arr, 25)
            # _median_num = np.median(arr)
            
            
            # a.view(a.shape[0], -1).min(1).values

        _mean.div_(len(img_list))
        _std .div_(len(img_list))
        _min.div_(len(img_list))
        _max.div_(len(img_list))
        mean_std = torch.stack([_mean,_std,_min,_max], dim=0).numpy() #[M, 2, N, K]
        _mean=_mean.numpy()
        _std =_std .numpy()
        _min =_min .numpy()
        _max =_max .numpy()
        np.savetxt(mean_file,mean_std,delimiter=",")
        # f.close()
        print("time elapsed: {} and saved mean and std to {}.." .format(time() - before,mean_file))
    print('dataset mean and std is [{}],[{}]'.format(_mean, _std))
    print('dataset min and max is [{}],[{}]'.format(_min, _max))
    return _mean,_std,_max,_min






def CT_loader(path_set=[],argument = False):
    imgs = []
    for path in path_set:
        img = sklearn_minmax(np.load(path))
        img = torch.Tensor(img)
        imgs.append(img)

    return imgs

def sklearn_minmax(x,max=1):
    scaler = MinMaxScaler()
    if x.ndim == 2:
        d3, d4 = x.shape  #
        x_train = x.reshape(-1, d4)
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_train = x_train.reshape(d3, d4) * max
    elif x.ndim == 3:
        d2, d3, d4 = x.shape  #
        x_train = x.reshape(-1, d4)
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_train = x_train.reshape(d2, d3, d4) * max
    elif x.ndim == 4:
        d1,d2, d3, d4 = x.shape  #
        x_train = x.reshape(-1, d4)
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_train = x_train.reshape(d1,d2, d3, d4) * max
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(x.ndim))
    return x_train


def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)


def make_dir_tree_from_file(file_path):
    import os
    directory = Path(os.path.dirname(file_path))
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(str(directory)+"has maked")
    return file_path


import random
def sigmoid_contrast(img,gain=15,cutoff=0.35,p=0.5):
    if random.random()>p:
        sigmoid_cutoff = lambda x: 1 / (1 + np.exp(gain * (cutoff - x)))
        return sigmoid_cutoff(img)
    else:
        return img

def img_rand_crop(img_L, img_H, window_size,min_H=0,min_W=0,max_H=0,max_W=0):
    # --------------------------------
    # randomly crop the L patch
    # crop corresponding H patch
    # --------------------------------
    ### todo rand crop at fixed row
    def smaller__(H,W,window_size,max_H,max_W,min_H,min_W):
        assert (max(0, H - window_size - max_H) - min_H) < 0, 'H_max-H_min({}-{}={}) is smaller than window{}'.format(max(0, H - max_H),min_H,max(0, H - window_size - max_H)-min_H,window_size)
        assert (max(0, W - window_size - max_W) - min_W) < 0, 'W_max-W_min({}-{}={}) is smaller than window{}'.format(max(0, H - max_H),min_H,max(0, H - window_size - max_H)-min_H,window_size)

    # if str(type(x)) == "<class 'numpy.ndarray'>":
    #     if len(np.shape(x)) < 3:
    #         x = np.expand_dims(x,0)
    # if str(type(x)) == "<class 'torch.Tensor'>":
    if str(type(img_L)) == "<class 'numpy.ndarray'>":
        if img_L.ndim == 2:
            H, W = img_L.shape
            # smaller__(H, W, window_size, max_H, max_W, min_H, min_W)
            rnd_h = random.randint(min_H, max(0, H - window_size - max_H))
            rnd_w = random.randint(min_W, max(0, W - window_size - max_W))
            img_L = img_L[rnd_h:rnd_h + window_size, rnd_w:rnd_w + window_size]
            img_H = img_H[rnd_h:rnd_h + window_size, rnd_w:rnd_w + window_size]
        elif img_L.ndim == 3:
            H, W, C = img_L.shape
            # smaller__(H, W, window_size, max_H, max_W, min_H, min_W)
            rnd_h = random.randint(min_H, max(0, H - window_size - max_H))
            rnd_w = random.randint(min_W, max(0, W - window_size - max_W))
            img_L = img_L[rnd_h:rnd_h + window_size, rnd_w:rnd_w + window_size, :]
            img_H = img_H[rnd_h:rnd_h + window_size, rnd_w:rnd_w + window_size, :]
        else:
            raise ValueError('Wrong img ndim: [{:d}].'.format(img_L.ndim))
    elif str(type(img_L)) == "<class 'torch.Tensor'>":
        if img_L.ndim == 2:
            H, W = img_L.shape
            # smaller__(H, W, window_size, max_H, max_W, min_H, min_W)
            rnd_h = random.randint(min_H, max(0, H - window_size - max_H))
            rnd_w = random.randint(min_W, max(0, W - window_size - max_W))
            img_L = img_L[rnd_h:rnd_h + window_size, rnd_w:rnd_w + window_size]
            img_H = img_H[rnd_h:rnd_h + window_size, rnd_w:rnd_w + window_size]
        elif img_L.ndim == 3:
            C, H, W = img_L.shape
            # smaller__(H, W, window_size, max_H, max_W, min_H, min_W)
            rnd_h = random.randint(min_H, max(0, H - window_size - max_H))
            rnd_w = random.randint(min_W, max(0, W - window_size - max_W))
            img_L = img_L[ :,rnd_h:rnd_h + window_size, rnd_w:rnd_w + window_size]
            img_H = img_H[ :,rnd_h:rnd_h + window_size, rnd_w:rnd_w + window_size]
        else:
            raise ValueError('Wrong img ndim: [{:d}].'.format(img_L.ndim))
    return img_L, img_H

# 0. 계산을 편하게 하기 위해 넘파이를 가져옵니다.
import numpy as np

class bin_totalScaler:
    # 1) 생성자에서 최대값, 최소값을 정의해줍니다.
    def __init__(self):
        self.max_num = -np.inf
        self.min_num = np.inf
        self.mean_num = None
        self.std_num = None
        self.q3 = None
        self.q1 = None
        self.median_num = None

    # 2) 최대값, 최소값을 계산해줍니다.
    def fit(self, arr):
        if arr is None:
            print("fit() missing 1 required positional argument: 'X'")

        self.max_num = np.min(arr)
        self.min_num = np.max(arr)
        self.mean_num = np.mean(arr)
        self.std_num = np.std(arr)

    # 3) 최대값, 최소값을 계산하며 동시에 scaled를 적용합니다.
    def fit_transform(self, arr):
        if arr is None:
            print("fit_transform() missing 1 required positional argument: 'X'")

        self.max_num = np.max(arr)
        self.min_num = np.min(arr)

        # MinMaxScaler 계산 공식을 적용합니다.
        return (arr - self.min_num) / (self.max_num - self.min_num)
    # 4) 이미 계산된 최대값, 최소값으로 scaled를 적용합니다.

    def transform(self, arr):
        return (arr - self.min_num) / (self.max_num - self.min_num)

    def transform_arr(self, arr):
        arrrr= []
        for idx,i in enumerate(arr):
            arrrr.append((i - self.min_num[idx]) / (self.max_num[idx] - self.min_num[idx]))
        return np.array(arrrr)

    def transform_std(self, arr,mean_num = None,std_num = None):
        if mean_num is not None and std_num is not None:
            return (arr - mean_num) / std_num
        else: return (arr - self.mean_num) / self.std_num

    def transform_arr_std(self, arr):
        arrrr= []
        for idx,i in enumerate(arr):
            arrrr.append((i - self.mean_num[idx]) / self.std_num[idx])
        return np.array(arrrr)



# 1. 클래스를 만들어줍니다.
class bin_MinMaxScaler:
    # 1) 생성자에서 최대값, 최소값을 정의해줍니다.
    def __init__(self):
        self.max_num = -np.inf
        self.min_num = np.inf
        self.mean_num = None
        self.std_num = None

    # 2) 최대값, 최소값을 계산해줍니다.
    def fit(self, arr):
        if arr is None:
            print("fit() missing 1 required positional argument: 'X'")

        self.max_num = np.min(arr)
        self.min_num = np.max(arr)

    # 3) 최대값, 최소값을 계산하며 동시에 scaled를 적용합니다.
    def fit_transform(self, arr):
        if arr is None:
            print("fit_transform() missing 1 required positional argument: 'X'")

        self.max_num = np.max(arr)
        self.min_num = np.min(arr)

        # MinMaxScaler 계산 공식을 적용합니다.
        return (arr - self.min_num) / (self.max_num - self.min_num)
    # 4) 이미 계산된 최대값, 최소값으로 scaled를 적용합니다.

    def transform(self, arr):
        return (arr - self.min_num) / (self.max_num - self.min_num)

    def transform_arr(self, arr):
        arrrr= []
        for idx,i in enumerate(arr):
            arrrr.append((i - self.min_num[idx]) / (self.max_num[idx] - self.min_num[idx]))
        return np.array(arrrr)

# 1. 클래스를 만들어줍니다.
class bin_StandardScaler:
    # 1) 생성자에서 평균, 표준편차를 정의해줍니다.
    def __init__(self):
        self.mean_num = None
        self.std_num = None

    # 2) 평균, 표준편차를 계산해줍니다.
    def fit(self, arr):
        if arr is None:
            print("fit() missing 1 required positional argument: 'X'")

        self.mean_num = np.mean(arr)
        self.std_num = np.std(arr)

    # 3) 평균, 표준편차를 계산하며 동시에 scaled를 적용합니다.
    def fit_transform(self, arr):
        if arr is None:
            print("fit_transform() missing 1 required positional argument: 'X'")

        self.mean_num = np.mean(arr)
        self.std_num = np.std(arr)

        # StandardScaler 계산 공식을 적용합니다.
        return (arr - self.mean_num) / self.std_num
    # 4) 이미 계산된 평균, 표준편차로 scaled를 적용합니다.
    def transform(self, arr):
        return (arr - self.mean_num) / self.std_num

class bin_RobustScaler:
    # 1) 생성자에서 q3, q1, 중앙값을 정의해줍니다.
    def __init__(self):
        self.q3 = None
        self.q1 = None
        self.median_num = None

    # 2) q3, q1, 중앙값을 계산해줍니다.
    def fit(self, arr):
        if arr is None:
            print("fit() missing 1 required positional argument: 'X'")

        self.q3 = np.percentile(arr, 75)
        self.q1 = np.percentile(arr, 25)
        self.median_num = np.median(arr)

    # 3) q3, q1, 중앙값을 계산하며 동시에 scaled를 적용합니다.
    def fit_transform(self, arr):
        if arr is None:
            print("fit_transform() missing 1 required positional argument: 'X'")

        self.q3 = np.percentile(arr, 75)
        self.q1 = np.percentile(arr, 25)
        self.median_num = np.median(arr)

        # RobustScaler 계산 공식을 적용합니다.
        return (arr - self.median_num) / (self.q3 - self.q1)

    # 4) 이미 계산된 q3, q1, 중앙값으로 scaled를 적용합니다.
    def transform(self, arr):
        return (arr - self.median_num) / (self.q3 - self.q1)


def clahe_(image,clip_limit=2,tile_grid=(8,8)):

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    clahe_image = clahe.apply((image*255).astype(np.uint8))

    return clahe_image


def clahe_on_rgb(image):

    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    height, width = red_channel.shape

    red_channel_clahe = clahe_(red_channel)
    green_channel_clahe = clahe_(green_channel)
    blue_channel_clahe = clahe_(blue_channel)

    clahe_image = np.zeros((height, width, 3), dtype=float)

    clahe_image[:, :, 0] = red_channel_clahe/255.0
    clahe_image[:, :, 1] = green_channel_clahe/255.0
    clahe_image[:, :, 2] = blue_channel_clahe/255.0

    return clahe_image

