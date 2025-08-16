import os
import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm

from train_gecko import DEVICE
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import batched_simil as mil


class GeckoDataset(Dataset):
    def __init__(self, features_deep_path, features_path, max_n_tokens=None):
        print(f'max_n_tokens:: {max_n_tokens}')
        self.max_n_tokens = max_n_tokens

        print('max number of tokens set to minimum of (dataset, fixed value of 2048)', self.max_n_tokens)
        self.features_deep_path = features_deep_path
        self.features_path = features_path
        self.slides = [file.split('.')[0] for file in os.listdir(features_deep_path)]

        self.feat_min, self.feat_max = self.__calculate_feat_stats()
        self.feats_size, self.feats_size_deep = self.__get_feat_sizes()
        assert len(self.slides) == len(
            os.listdir(features_path)), 'Number of deep features and concept priors do not match!'

    def __len__(self):
        return len(self.slides)

    def __get_feat_sizes(self):
        feat = np.array(pd.read_csv(f'{self.features_path}/{self.slides[0]}.csv'))
        deep_feat = self.__read_h5(self.slides[0])

        return feat.shape[1], deep_feat.shape[1]

    def __calculate_feat_stats(self):
        print('calculating global concept features stats')
        feat_array_min = []
        feat_array_max = []
        pbar = tqdm(self.slides, total=len(self.slides))
        for _, slide in enumerate(pbar):
            feat = np.array(pd.read_csv(f'{self.features_path}/{slide}.csv'))
            feat_array_min.append(feat.min(axis=0))
            feat_array_max.append(feat.max(axis=0))

        feat_min = np.stack(feat_array_min).min(axis=0)
        feat_max = np.stack(feat_array_max).max(axis=0)
        print('Finished calculating global concept features stats')

        return torch.tensor(feat_min, dtype=torch.float32), torch.tensor(feat_max, dtype=torch.float32)

    def __read_h5(self, slide_id):
        with h5py.File(f'{self.features_deep_path}/{slide_id}.h5', "r") as h5_file:
            embeddings_np = h5_file["features"][:]
            embedding = torch.tensor(embeddings_np, dtype=torch.float32)
            return embedding

    def __normalize_feature(self, feat, delta=1e-6):
        feat_norm = (feat - self.feat_min) / (self.feat_max - self.feat_min + delta)
        return feat_norm

    def __getitem__(self, idx):
        slide_id = self.slides[idx]
        bag_feats_deep = self.__read_h5(slide_id)
        # bag_feats_deep = bag_feats_deep.view(-1, self.feats_size_deep)

        # bag_feats = pd.read_csv(f'{self.features_path}/{slide_id}.csv')
        bag_feats = torch.tensor(np.array(pd.read_csv(f'{self.features_path}/{slide_id}.csv')), dtype=torch.float32)
        # bag_feats = bag_feats.view(-1, self.feats_size)

        bag_feats = self.__normalize_feature(bag_feats)

        if bag_feats.shape[0] > self.max_n_tokens:
            patch_indices = np.random.choice(np.arange(bag_feats.shape[0]), self.max_n_tokens, replace=False)
        elif bag_feats.shape[0] == self.max_n_tokens:
            patch_indices = np.arange(bag_feats.shape[0])
        else:
            patch_indices = np.concatenate([np.arange(bag_feats.shape[0]),
                                            np.random.choice(np.arange(bag_feats.shape[0]),
                                                             self.max_n_tokens - bag_feats.shape[0])])

        bag_feats_deep = bag_feats_deep[patch_indices].to(DEVICE)
        bag_feats = bag_feats[patch_indices].to(DEVICE)

        return slide_id, bag_feats_deep, bag_feats


def extract(ssl_model, dataloader):
    ssl_model.eval()

    labels_list = []
    bag_features_dict = dict()
    bag_features_deep_dict = dict()
    attention_test_bag_patch = dict()
    attention_test_bag_feature = dict()

    with torch.no_grad():

        for i, (slide_id, patch_emb_deep, patch_emb) in enumerate(tqdm(dataloader)):

            bag_features, _, _, bag_features_deep, A_feat, A_patch = ssl_model(patch_emb, patch_emb_deep)

            # labels_list.append(int(bag_label[0]))

            bag_features_dict[slide_id[0]] = bag_features.squeeze(0).clone().cpu().detach()
            bag_features_deep_dict[slide_id[0]] = bag_features_deep.squeeze(0).clone().cpu().detach()

            attention_test_bag_patch[slide_id[0]] = A_patch.squeeze(0).squeeze(-1).clone().cpu().detach()
            attention_test_bag_feature[slide_id[0]] = A_feat.squeeze(0).clone().cpu().detach()

    # labels_list = np.array(labels_list)

    return bag_features_dict, bag_features_deep_dict, attention_test_bag_patch, attention_test_bag_feature


def infer(features_deep_path, features_path, max_n_tokens, model_weights_path, out_path, top_k=10):
    dataset = GeckoDataset(
        features_deep_path=features_deep_path,
        features_path=features_path,
        max_n_tokens=max_n_tokens,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    b_classifier = mil.BClassifier(input_size=dataset.feats_size, input_size_deep=dataset.feats_size_deep,
                                   top_k=top_k).to(DEVICE)
    ssl_model = mil.MILNet(b_classifier).to(DEVICE)

    state_dict_weights = torch.load(model_weights_path)
    msg = ssl_model.load_state_dict(state_dict_weights, strict=False)
    print(msg)

    (bag_features_dict, bag_features_deep_dict, attention_test_bag_patch,
     attention_test_bag_feature) = extract(ssl_model, dataloader)

    slides = bag_features_deep_dict.keys()
    pbar = tqdm(slides, total=len(slides))
    os.makedirs(out_path, exist_ok=True)

    for i,slide_id in enumerate(pbar):
        h5_filename = f'{out_path}/{slide_id}.h5'
        with h5py.File(h5_filename, 'w') as f:
            f.create_dataset('bag_feats', data=bag_features_dict[slide_id])
            f.create_dataset('bag_feats_deep', data=bag_features_deep_dict[slide_id])
            f.create_dataset('attention_bag_patches', data=attention_test_bag_patch[slide_id])
            f.create_dataset('attention_bag_feats', data=attention_test_bag_feature[slide_id])

    print('Finished writing features!')

if __name__=='__main__':
    features_deep_path = '/mnt/saarthak/datasets/REG_test2_revised_processed/20x_512px_0px_overlap/features_conch_v1'
    features_path = '/mnt/saarthak/datasets/REG_test2_revised_processed/20x_512px_0px_overlap/concept_prior_conch_v1_with_organ_context'

    # features_deep_path = '/mnt/surya/projects/GECKO/test_data/test_feat_deep'
    # features_path = '/mnt/surya/projects/GECKO/test_data/test_feat_con'
    # out_path = '/mnt/surya/projects/GECKO/test_data/output'
    out_path = '/mnt/surya/dataset/REG_2025/gecko_t_woc'
    max_n_tokens = 2048
    model_weights_path = 'exp_t_woc_1/_keepratio0.7/topk10_mintokensize512_maxtokensize2048_lr0.0001_epochs50_bs128_temperatureNCE0.01/0/checkpoint.pth'
    infer(features_deep_path, features_path, max_n_tokens, model_weights_path, out_path)

