import h5py
import sys
import os
import numpy as np
import torch

import torch
from PIL import Image
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse
import torch.nn.functional as F

sys.path.insert(1, 'GECKO/data_curation/CONCH')

from conch.open_clip_custom import create_model_from_pretrained

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, preprocess = create_model_from_pretrained('conch_ViT-B-16', 'CONCH/checkpoints/conch/pytorch_model.bin', device=device)
_ = model.eval()

save_path = '/mnt/saarthak/datasets/REG_test2_revised_processed/20x_512px_0px_overlap/features_conch_v1_cosine_sim_proj'

for path in tqdm(sorted(os.listdir('/mnt/saarthak/datasets/REG_test2_revised_processed/20x_512px_0px_overlap/features_conch_v1'))[:]):
    with h5py.File(f'/mnt/saarthak/datasets/REG_test2_revised_processed/20x_512px_0px_overlap/features_conch_v1/{path}', 'r') as f:
        features = f['features'][:]
        pooled = F.normalize(torch.Tensor(features).cuda() @ model.visual.proj_contrast, dim=-1)
        torch.save(pooled, f'{save_path}/{path.replace(".h5", ".pt")}')
