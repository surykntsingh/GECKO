import numpy as np
import torch
from tqdm import tqdm

from train_gecko import GeckoDataset, DEVICE
from torch.utils.data import DataLoader
import batched_simil as mil


def extract(ssl_model, dataloader):
    ssl_model.eval()

    labels_list = []
    bag_features_dict = dict()
    bag_features_deep_dict = dict()
    attention_test_bag_patch = dict()
    attention_test_bag_feature = dict()

    with torch.no_grad():

        for i, (patch_emb_deep, patch_emb) in enumerate(tqdm(dataloader)):

            bag_features, _, _, bag_features_deep, A_feat, A_patch = ssl_model(patch_emb, patch_emb_deep)

            # labels_list.append(int(bag_label[0]))

            bag_features_dict[i] = bag_features.squeeze(0).clone().cpu().detach()
            bag_features_deep_dict[i] = bag_features_deep.squeeze(0).clone().cpu().detach()

            attention_test_bag_patch[i] = A_patch.squeeze(0).squeeze(-1).clone().cpu().detach()
            attention_test_bag_feature[i] = A_feat.squeeze(0).clone().cpu().detach()

    # labels_list = np.array(labels_list)

    return bag_features_dict, bag_features_deep_dict, attention_test_bag_patch, attention_test_bag_feature


def infer(features_deep_path, features_path, max_n_tokens, model_weights_path, top_k=10):
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

    print((bag_features_dict, bag_features_deep_dict, attention_test_bag_patch,
     attention_test_bag_feature))

if __name__=='__main__':
    # features_deep_path = '/mnt/saarthak/datasets/REG_processed/20x_512px_0px_overlap/features_conch_v1'
    # features_path = '/mnt/saarthak/datasets/REG_processed/20x_512px_0px_overlap/concept_prior_conch_v1'

    features_deep_path = '/mnt/surya/projects/GECKO/test_data/test_feat_deep'
    features_path = '/mnt/surya/projects/GECKO/test_data/test_feat_con'
    max_n_tokens = 2048
    model_weights_path = 'exp_2/_keepratio0.7/topk10_mintokensize512_maxtokensize2048_lr0.0001_epochs10_bs128_temperatureNCE0.01/0/checkpoint.pth'
    infer(features_deep_path, features_path, max_n_tokens, model_weights_path)
