import argparse
from haven import haven_utils as hu
from haven import haven_wizard as hw
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
import soft_renderer as sr
import copy
import debug
from src import losses
from src import datasets
from src import models
from pytorch3d.structures import Meshes
import time, tqdm
import os



class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}


if __name__ == "__main__":
    import exp_configs
    cl_dict = {}
    for c in class_ids_map.keys():
        cl_dict[c] = 0.
        for split in ['train', 'val', 'test']:
            cl_dict[c] += len(list(np.load(os.path.join('/mnt/public/datasets2/softras/mesh_reconstruction', 
                    '%s_%s_voxels.npz' % (c, 
                    split))).items())[0][1])
    # print(cl_dict)
    for r, v in cl_dict.items():
        print(f'{int(v)} &', end =" ") 
    # print(f'{r["test_iou (max)"]/100:.2f} &', end =" ") 
