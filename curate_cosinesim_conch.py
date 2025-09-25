import torch
import sys
import json
import torch
import pandas as pd
import pickle
import argparse
from torch.nn import functional as F
from tqdm import tqdm
sys.path.insert(1, 'CONCH')
from conch.open_clip_custom import tokenize, get_tokenizer
from conch.open_clip_custom import create_model_from_pretrained
import os

tokenizer = get_tokenizer()

parser = argparse.ArgumentParser()
parser.add_argument('--image_feat_path', type=str, default = "/mnt/surya/dataset/sbu/processed_features/20x_512px_0px_overlap/features_conch_v1_cosine_sim_proj")
parser.add_argument('--patch_prompts_path', type=str, default = "prompts/patch_level_prompts.json")
parser.add_argument('--save_path', type=str, default = '/mnt/surya/dataset/sbu/processed_features/20x_512px_0px_overlap/concept_prior_conch_v1_with_organ_context')
parser.add_argument('--conch_model_path', type=str, default = '../CONCH/checkpoints/conch/pytorch_model.bin')
parser.add_argument('--conch_model_type', type=str, default = 'conch_ViT-B-16')
args = parser.parse_args()

image_feat_path = args.image_feat_path
patch_prompts_path = args.patch_prompts_path
save_path = args.save_path


model, preprocess = create_model_from_pretrained(args.conch_model_type, args.conch_model_path, device='cuda')
_ = model.eval()

with open(patch_prompts_path, 'r', encoding='utf-8') as file:
    patch_level_prompts = json.load(file)

patch_level_prompts_ = []
for patch_level_prompt in list(patch_level_prompts.keys()):
    for patch_level_prompt_i in patch_level_prompts[patch_level_prompt]:
        patch_level_prompts_.append(f"{patch_level_prompt_i} in {patch_level_prompt}; {patch_level_prompts[patch_level_prompt][patch_level_prompt_i]}")

patch_level_prompts_ = [name.replace("_", " ") for name in patch_level_prompts_]
patch_level_prompts_ = [name[:-1] + ";" if name[-1] == '.' else name + ";" for name in patch_level_prompts_]


print('len(patch_level_prompts_)', len(patch_level_prompts_))

print(patch_level_prompts_)



patch_text_features = []
for classname in patch_level_prompts_:

    texts = [classname]
    token_ids = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
    token_ids = token_ids.to('cuda')

    with torch.no_grad():
        classname_embeddings = F.normalize(model.encode_text(token_ids), dim=-1)[0]

    patch_text_features.append(classname_embeddings)

patch_text_features = torch.stack(patch_text_features, dim=0).cpu().detach()


print('patch_text_features.shape', patch_text_features.shape)

for path in tqdm(sorted(os.listdir(image_feat_path))[:]):
    patch_features = torch.load(image_feat_path + '/' + path).cpu().detach()  # already norm values through CONCH
    # print('patch_features.shape', patch_features.shape)

    sim_matrix_raw_concept_df = pd.DataFrame(patch_features @ patch_text_features.t(), columns = patch_level_prompts_)
    sim_matrix_raw_concept_df.to_csv(save_path + '/' + path.replace(".pt", ".csv"), index=False)
