import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logging import *
from models.ObitoNet import ObitoNet
import matplotlib.pyplot as plt

import cv2
import numpy as np


def test_net(args, config):
    logger = None
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    # base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)

    # Build Point Cloud Encoder
    obitonet_pc = builder.obitonet_pc_builder(config.model)
    #<TODO> Build Image Encoder
    obitonet_img = builder.obitonet_img_builder(config.model)
    # Build Cross Attention Decoder
    obitonet_ca = builder.obitonet_ca_builder(config.model)
    # Build ObitoNet
    obitonet = ObitoNet(config.model, obitonet_pc, obitonet_img, obitonet_ca)

    # builder.load_model(base_model, args.ckpts, logger = logger)
    builder.load_model(obitonet_pc, 'ObitoNetPC', args, logger = logger)
    builder.load_model(obitonet_img, 'ObitoNetImg', args, logger = logger)
    builder.load_model(obitonet_ca, 'ObitoNetCA', args, logger = logger)

    if args.use_gpu:
        obitonet.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(obitonet, test_dataloader, args, config, logger=logger)

def save_attention_heatmap(attention_matrix, save_path):
    """
    Save the attention matrix as a heatmap.
    Args:
        attention_matrix (np.array): Attention weights (B, num_heads, N, M).
        save_path (str): Path to save the heatmap.
    """
    # Average over heads if needed
    avg_attention = attention_matrix.mean(axis=1)  # Shape: (B, N, M)
    
    # Visualize the first example
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_attention[0], cmap='viridis')
    plt.colorbar()
    plt.title("Cross Attention Weights")
    plt.xlabel("Keys (from y)")
    plt.ylabel("Queries (from x)")
    plt.savefig(save_path)
    plt.close()

# visualization
def test(obitonet, test_dataloader, args, config, logger = None):

    obitonet.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156", #plane
        "04379243",  #table
        "03790512", #motorbike
        "03948459", #pistol
        "03642806", #laptop
        "03467517",     #guitar
        "03261776", #earphone
        "03001627", #chair
        "02958343", #car
        "04090263", #rifle
        "03759954", # microphone
    ]
    with torch.no_grad():
        for idx, (data, img) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            # if  taxonomy_ids[0] not in useful_cate:
            #     continue
            # if taxonomy_ids[0] == "02691156":
            #     a, b= 90, 135
            # elif taxonomy_ids[0] == "04379243":
            #     a, b = 30, 30
            # elif taxonomy_ids[0] == "03642806":
            #     a, b = 30, -45
            # elif taxonomy_ids[0] == "03467517":
            #     a, b = 0, 90
            # elif taxonomy_ids[0] == "03261776":
            #     a, b = 0, 75
            # elif taxonomy_ids[0] == "03001627":
            #     a, b = 30, -45
            # else:
            a, b = 0, 0


            dataset_name = config.dataset.test._base_.NAME
            # if dataset_name == 'ShapeNet':
            points = data.cuda()
            img = img.cuda()
            # else:
            #     raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # dense_points, vis_points = base_model(points, vis=True)
            dense_points, vis_points, centers, attention_matrix = obitonet(points, img, vis=True)

            final_image = []
            data_path = f'./vis/'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points,a,b)
            final_image.append(points[150:650,150:675,:])

            # centers = centers.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
            # centers = misc.get_ptcloud_img(centers)
            # final_image.append(centers)

            vis_points = vis_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            vis_points = misc.get_ptcloud_img(vis_points,a,b)

            final_image.append(vis_points[150:650,150:675,:])

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points,a,b)
            final_image.append(dense_points[150:650,150:675,:])

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot_{idx:04d}.jpg')
            cv2.imwrite(img_path, img)
            save_attention_heatmap(
                attention_matrix.cpu().numpy(),
                os.path.join(data_path, f'attention_{idx:04d}.png')
            )

            if idx > 1500:
                break

        return