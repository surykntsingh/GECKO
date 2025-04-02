slide_dir="$1"
patch_dir="$2"
lists_feats_dir="$3"
conch_model_path="$4"
patch_prompts="$5"


# Run deepzoom tiler
python deepzoom_tiler_organ.py --dataset "$slide_dir" --save_path "$patch_dir" --workers 10 --magnifications 1 --tile_size 448 --background_t 15

# # Generate patch dictionary list
python patch_dict_list.py --patch_path "$patch_dir" --save_path "$lists_feats_dir"

# Extract conch features
python extract_conch_feat.py --img_list_path "$lists_feats_dir/all_list.pickle" --save_path "$lists_feats_dir" --conch_model_path "$conch_model_path"

# Curate cosine similarity using conch features
python curate_cosinesim_conch.py --img_list_path "$lists_feats_dir/all_list.pickle" --img_dict_path "$lists_feats_dir/all_dict.pickle" --image_feat_path "$lists_feats_dir/deep_features_for_cosine_sim.pth" --patch_prompts_path "$patch_prompts" --save_path "$lists_feats_dir" --conch_model_path "$conch_model_path"
