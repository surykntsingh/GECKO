import h5py
import os
import torch
import sys
import torch
from tqdm import tqdm

import argparse
import torch.nn.functional as F
parser = argparse.ArgumentParser()
sys.path.insert(1, 'data_curation/CONCH')

from conch.open_clip_custom import create_model_from_pretrained

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default = "/mnt/surya/dataset/sbu/processed_features/20x_512px_0px_overlap/features_conch_v1_cosine_sim_proj")
parser.add_argument('--feature_dir', type=str, default = "/mnt/surya/dataset/sbu/processed_features/20x_512px_0px_overlap/features_conch_v1")
parser.add_argument('--conch_model_type', type=str, default = 'conch_ViT-B-16')
parser.add_argument('--conch_model_path', type=str, default = '../CONCH/checkpoints/conch/pytorch_model.bin')

args = parser.parse_args()


model, preprocess = create_model_from_pretrained('conch_ViT-B-16', args.conch_model_path, device=device)
_ = model.eval()

for path in tqdm(sorted(os.listdir(args.feature_dir))[:]):
    with h5py.File(f'/{args.feature_dir}/{path}', 'r') as f:
        features = f['features'][:]
        pooled = F.normalize(torch.Tensor(features).cuda() @ model.visual.proj_contrast, dim=-1)
        torch.save(pooled, f'{args.save_path}/{path.replace(".h5", ".pt")}')
