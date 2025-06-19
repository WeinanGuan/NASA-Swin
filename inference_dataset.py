import os
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import *
from model import Detector
from models.RIDNet.ridnet import RIDNET


transforms_dict={
    'ImageNet': {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]},
    'Xception': {'mean':[0.5, 0.5, 0.5], 'std':[0.5, 0.5, 0.5]},
    'genernal': {'mean':[0, 0, 0], 'std':[1, 1, 1]}
}


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def loss_cal(output_list, target_list):
    criterion=torch.nn.NLLLoss()

    fake_pred=torch.tensor(output_list)
    real_pred=1-fake_pred
    pred=torch.stack([real_pred,fake_pred], dim=1)

    target=torch.tensor(target_list).long()
    
    loss=criterion(torch.log(pred),target)

    return loss


def main(args):
    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name, map_location='cpu')["model"]
    model.net.load_state_dict({k.replace('module.', ''): v for k, v in cnn_sd.items()})
    model.eval()

    RIDNet=RIDNET(n_feats=64, reduction=16, rgb_range=255)
    RIDNet=RIDNet.to(device)
    RIDNet.load_state_dict(torch.load('./models/RIDNet/experiment/ridnet.pt', map_location='cpu'))
    RIDNet.eval()

    test_dataset=TestDataset(image_size=args.image_size, dataset=args.dataset, data_root=args.data_root)
    test_dataloader=DataLoader(dataset=test_dataset,
                               batch_size=args.batch_size,
                               shuffle=False,
                               drop_last=False)
    
    model_realoutput_list=[]
    model_fakeoutput_list=[]
    output_list=[]
    target_list=[]
    pbar=tqdm(test_dataloader)
    for step, data in enumerate(pbar):
        image, label=data
        image=image.to(device, non_blocking=True).float()
        label_list=label.tolist()
        target_list+=label_list

        with torch.no_grad():
            RID_output=RIDNet(image)
            RID_output=quantize(RID_output, 255)
            
            diff=image-RID_output
            diff=(diff+255)/(2*255)
            
            x_res=transforms.Normalize(transforms_dict[args.transformation]['mean'], transforms_dict[args.transformation]['std'])(diff)
            x_rgb=transforms.Normalize(transforms_dict[args.transformation]['mean'], transforms_dict[args.transformation]['std'])(image/255)
            model_pred=model.forward_RIDRes(x_res=x_res, x_rgb=x_rgb)
            pred=model_pred.softmax(1)[:,1]

        output_list+=pred.cpu().tolist()
        model_realoutput_list+=model_pred[:,0].cpu().tolist()
        model_fakeoutput_list+=model_pred[:,1].cpu().tolist()
    
    acc=np.equal(np.array(target_list), np.array(output_list)>0.5).sum()/len(target_list)
    loss=loss_cal(output_list, target_list)
    print(f'{args.dataset}| Acc: {acc:.4f}| LogLoss: {loss:.4f}')

    real_target = [target_list[inx] for inx in range(len(target_list)) if target_list[inx]==0]
    real_output = [output_list[inx] for inx in range(len(target_list)) if target_list[inx]==0]
    real_acc = np.equal(np.array(real_target), np.array(real_output)>0.5).sum()/len(real_target)
    real_loss = loss_cal(real_output, real_target)

    fake_target = [target_list[inx] for inx in range(len(target_list)) if target_list[inx]==1]
    fake_output = [output_list[inx] for inx in range(len(target_list)) if target_list[inx]==1]
    fake_acc = np.equal(np.array(fake_target), np.array(fake_output)>0.5).sum()/len(fake_target)
    fake_loss = loss_cal(fake_output, fake_target)
    print(f'{args.dataset}-Real| Acc: {real_acc:.4f}| LogLoss: {real_loss:.4f}')
    print(f'{args.dataset}-Fake| Acc: {fake_acc:.4f}| LogLoss: {fake_loss:.4f}')
    print('-'*50)


if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-r',dest='data_root',type=str)
    parser.add_argument('-d',dest='dataset',type=str, choices=['stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'ADM', 'glide', 'Midjourney', 'VQDM', 'wukong', 'BigGAN'])
    parser.add_argument('-b',dest='batch_size', default=64, type=int)
    parser.add_argument('-s',dest='image_size', default=224, type=int)
    parser.add_argument('-t',dest='transformation', default='ImageNet', choices=['ImageNet', 'Xception', 'general'])
    args=parser.parse_args()

    print('-'*50)
    print(args)
    print('-'*50)

    main(args)
