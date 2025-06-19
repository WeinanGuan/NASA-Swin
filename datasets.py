import torch
from torchvision import transforms
from glob import glob
import cv2
from torch.utils.data import Dataset
import random, os


def GenImage(data_root, diffusion_model):
    data_dir = os.path.join(data_root, f'{diffusion_model}/*/val')
    
    real_image_list=list(sorted(glob(data_dir+'/nature/*')))
    fake_image_list=list(sorted(glob(data_dir+'/ai/*')))

    image_list=real_image_list+fake_image_list
    label_list=[0]*len(real_image_list)+[1]*len(fake_image_list)

    return image_list, label_list


class TestDataset(Dataset):
    def __init__(self, image_size=224, dataset='stable_diffusion_v_1_4', data_root=''):
        image_list, label_list=GenImage(data_root=data_root, diffusion_model=dataset)

        self.image_list=image_list
        self.label_list=label_list

        self.image_size=(image_size, image_size)

        self.transform=self.get_transforms()
        print(f'{dataset}-Dataset-test: {len(image_list)}\nReal:{label_list.count(0)}\nFake:{len(image_list)-label_list.count(0)}')


    def __len__(self):
        return len(self.image_list)
    

    def __getitem__(self, index):
        flag=True
        while flag:
            try:
                image_path = self.image_list[index]
                image_label = self.label_list[index]

                image=cv2.imread(image_path)
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image=self.CenterCrop(image)

                image=self.transform(image).float()
                flag=False
            except Exception as e:
                print(e, image_path)
                index=torch.randint(low=0,high=len(self),size=(1,)).item()

        return image, image_label
    

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.PILToTensor()
        ])
    

    def CenterCrop(self, image):
        if min(image.shape[:2])>512:
            scale=512/min(image.shape[:2])
            image=cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h,w=image.shape[:2]
        side_length=min(h,w,224)

        center_h, center_w=h/2, w/2

        h0=max(int(center_h-side_length/2),0)
        h1=min(int(center_h+side_length/2),h)
        w0=max(int(center_w-side_length/2),0)
        w1=min(int(center_w+side_length/2),w)

        return image[h0:h1, w0:w1]