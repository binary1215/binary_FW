import torch
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np
from glob import glob
from os.path import join
from PIL import Image

def label(input_list):
    label_list = []
    for path in input_list:
        label_list.append(int((path.split('.')[-2])[-1:]))  #각 이미지별라벨 추출
    # unique_label_names = np.unique(label_list)              #라벨 별 중복값 제거하고 총 라벨 호출
    # label_array = np.tile(np.array(label_list), 10).reshape(-1, label_list.__len__())   #라벨리스트를 0,1,2,3,4,5,6,7,8,9 *10 으로 변환
    # onehot_label = (label_array.T == unique_label_names).astype(np.long)               #라벨 어레이 에서 트랜스폼하여 float32로 변환 하여 겹치는것만 참값으로 받고, 참값을 int로 받으면 원핫인코딩 완성
    return label_list

# transform = T.Compose([T.ToTensor()])
transform = T.Compose(
    [T.ToTensor(),
    # torch.clamp(,min=0,max=1)
     T.Normalize([0.1307], [0.3081])])
# transform = T.Compose(
#     [T.ToTensor(),
#      T.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))])



class Dataset(data.Dataset):

    def __init__(self, root):

        self.input_list = sorted(glob(join(root, '*.png')))
        self.label_list = label(self.input_list)


    def __getitem__(self, index):

        return transform(Image.open(self.input_list[index])),\
               torch.tensor(self.label_list[index])

    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.input_list.__len__()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # write test codes to verify your implementations
    root = r'data/train'
    input_list = sorted(glob(join(root, '*.png')))
    print(input_list[1].find('.'))
    label =label(input_list)

    idx = 0
    image = Image.open(input_list[idx])
    np.array(image).shape #=(28,28)
    print(image)
    img_tensor = transform(image)
    print(img_tensor)
    img_tensor.shape    #(1,28,28)
    print(image)
    print(img_tensor.shape)
    img_array = np.array(img_tensor)
    img_array = np.transpose(img_array, [1,2,0])

    img_array = np.tile(image, 3)
    plt.imshow(image)       #succeed 원본

    dataloader = torch.utils.data.DataLoader(dataset=Dataset('data/train'),
                                              batch_size=10,
                                              shuffle=False)
    input,label = next(iter(dataloader))
    label = (dataloader)
    type(input) # <class 'torch.Tensor'>
    input.shape #torch.Size([10, 1, 28, 28])
    type(label) # <class 'torch.Tensor'>
    label.shape
    label

    #
    # idx = 0
    # input_list[idx]
    # image = Image.open(input_list[idx])
    # np.array(image).shape #=(28,28)
    # img_array = np.tile(image, (3,1,1))
    # img_array = np.transpose(img_array, [1, 2, 0])
    # np.array(img_array).shape
    # img_tensor = transform(img_array)
    # print(img_tensor)
    # img_tensor.shape    #(1,28,28)
    # print(img_tensor.shape)
    # img_array = np.array(img_tensor)
    # # img_array = np.transpose(img_array, [1,2,0])
    #
    # # img_array = np.tile(img_array, 3)
    # img_array = np.transpose(img_array, [2, 1, 0])
    # print(img_array.shape)
    # plt.imshow(img_array)       #succeed 수정본 성공
