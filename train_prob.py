import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

import os
import math
import random
from glob import glob
import os.path as osp
from PIL import Image

from pwcnet_prob import PWCDCNet_old, PWCDCNet
import cv2
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

"""
class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='data_scene_flow'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
"""

class KITTI(torch.utils.data.Dataset):
    def __init__(self):
        self.left_images = sorted(glob(os.path.join("./data_scene_flow/training", 'image_2/*_10.png')))
        self.right_images = sorted(glob(os.path.join("./data_scene_flow/training", 'image_2/*_11.png')))
        self.target_flows = sorted(glob(os.path.join("./data_scene_flow/training", 'flow_occ/*_10.png')))

    def __getitem__(self, index):
        left_image = Image.open(self.left_images[index])
        right_image = Image.open(self.right_images[index])
        
        target_flows, valids = self.readFlowGroundTruth(self.target_flows[index])
        target_flows = torch.from_numpy(target_flows)
        target_flows = target_flows.permute(2, 0, 1)
        target_flows = torchvision.transforms.CenterCrop(size = (320, 896))(target_flows)
        valids = torch.from_numpy(valids)
        valids = torchvision.transforms.CenterCrop(size = (320, 896))(valids)
        
        left_image = torchvision.transforms.CenterCrop(size = (320, 896))(left_image)
        left_image = torchvision.transforms.ToTensor()(left_image)
        
        right_image = torchvision.transforms.CenterCrop(size = (320, 896))(right_image)
        right_image = torchvision.transforms.ToTensor()(right_image)
        return left_image, right_image, torch.div(target_flows, 20), valids

    def __len__(self):
        return len(self.left_images)

    def readFlowGroundTruth(self, filename):
        flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        flow = flow[:, :, ::-1].astype(np.float32)
        flow, valid = flow[:, :, :2], flow[:, :, 2]
        flow = (flow - 2**15) / 64.0
        return flow, valid

def ProbabilisticNLLLoss(predicted_flows, target_flow, valids):
    # predicted_flows (flow2, flow3, flow4, flow5, flow6)
    upsample = torch.nn.Upsample(size=(320, 896), mode='bilinear', align_corners=False)
    loss = 0.0
    epsilon = 0.00001
    for predicted_flow in predicted_flows:
        upsampled_flow = upsample(predicted_flow)
        upsampled_flow = upsampled_flow * (320 // predicted_flow.shape[2])
        mean_u = upsampled_flow[:, 0, :, :]
        mean_v = upsampled_flow[:, 1, :, :]
        sigma_u = upsampled_flow[:, 2, :, :]
        sigma_v = upsampled_flow[:, 3, :, :]
        dist_u = torch.distributions.normal.Normal(mean_u, sigma_u)
        dist_v = torch.distributions.normal.Normal(mean_v, sigma_v)
        loss_u = -dist_u.log_prob(target_flow[:, 0, :, :])
        loss_v = -dist_v.log_prob(target_flow[:, 1, :, :])
        loss = loss + torch.mul(loss_u, valids).mean() + torch.mul(loss_v, valids).mean()
        """loss_u = torch.log(sigma_u + epsilon) + torch.div(torch.mul((mean_u - target_flow[:, 0, :, :]), (mean_u - target_flow[:, 0, :, :])), torch.mul(torch.mul(sigma_u, sigma_u), 2) + epsilon)
        loss_v = torch.log(sigma_v + epsilon) + torch.div(torch.mul((mean_v - target_flow[:, 1, :, :]), (mean_v - target_flow[:, 1, :, :])), torch.mul(torch.mul(sigma_v, sigma_v), 2) + epsilon)
        loss = loss + torch.mul(torch.sum(loss_u + loss_v), valids).mean()"""
    return loss

def upsample_flow_predictions(flow_list):
    # Shape of each flow: (batch_size, 2, Height, Width)
    upsample = torch.nn.Upsample(size=(320, 896), mode='bilinear', align_corners=False)
    upsampled_flows = []
    for flow in flow_list:
        upsampled_flow = upsample(flow)
        upsampled_flow = upsampled_flow * (320 // flow.shape[2])
        upsampled_flows.append(upsampled_flow)
    return upsampled_flows

def EPE(input_flow, target_flows, valids):
    # return torch.norm(target_flow-input_flow,p=2,dim=1).mean()
    loss = 0.0
    for idx in range(input_flow.shape[0]):
        u_diff = (input_flow[idx][0] - target_flows[idx][0])
        v_diff = (input_flow[idx][1] - target_flows[idx][1])
        mask = valids[idx]
        loss = loss + torch.mul(torch.sqrt(torch.add(torch.mul(u_diff, u_diff), torch.mul(v_diff, v_diff))), mask).mean()
    return loss

def total_EPE_loss(upsampled_flow_list, target_flows, valids):
    total_loss = 0.0
    for upsampled_flow in upsampled_flow_list:
        total_loss = total_loss + EPE(upsampled_flow, target_flows, valids)
    return total_loss

import wandb
wandb.init(project="pwcnet")
def main():
    # print(ProbabilisticNLLLoss((torch.rand(4,4, 20, 56), torch.rand(4, 4, 20, 56)), torch.rand(4, 2, 320, 896)))
    train_loader = torch.utils.data.DataLoader(KITTI(), batch_size = 4, shuffle = True)
    model = PWCDCNet()
    # model.load_state_dict(torch.load("./models/not_divided_by_20.pth"))
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    wandb.watch(model)
    for epoch in range(1, 200):
        running_loss = 0.0
        print("Epoch {} losses:".format(epoch))
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            left_image, right_image, target_flows, valids = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            # Shapes:
            ## Left Image: (batch_size, 3, 320, 896)
            ## Right Image: (batch_size, 3, 320, 896)
            ## Target Flows: (batch_size, 2, 320, 896)
            ## Valids: (batch_size, 320, 896)
            flow2, flow3, flow4, flow5, flow6 = model(torch.cat((left_image, right_image), 1))
            #upsampled_flows = upsample_flow_predictions([flow2[:, :2, :, :], flow3[:, :2, :, :], flow4[:, :2, :, :], flow5[:, :2, :, :], flow6[:, :2, :, :]])
            #loss = total_EPE_loss(upsampled_flows, target_flows, valids)
            loss = ProbabilisticNLLLoss([flow2, flow3, flow4, flow5, flow6], target_flows, valids)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print("[{} {}] loss: {:.5f}".format(epoch, i, running_loss / 10))
                wandb.log({"Epoch Number": epoch, "Itertaion Number": i, "Loss": running_loss / 10})
                running_loss = 0.0
        print("Epoch {} ended".format(epoch))
    torch.save(model.state_dict(), './models/prob_lr001_libloss_elu2_flow2_included.pth')
if __name__ == "__main__":
    main()