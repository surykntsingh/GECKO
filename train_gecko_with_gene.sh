
# Assign command-line arguments to variables
features_dir="$1"
experiment_dir="$2"
split_path="$3"
gene_expression_path="$4"


python train_gecko_with_gene.py --keep_ratio 0.7 --top_k 10 --cross_val_fold 0 --dataset_dict_path "$features_dir/all_dict.pickle" --features_deep_path "$features_dir/deep_features.pth" --features_path "$features_dir/concept_prior.csv" --save_path "$experiment_dir" --split_path "$split_path" --gene_expression_path "$gene_expression_path" 

python train_gecko_with_gene.py --keep_ratio 0.7 --top_k 10 --cross_val_fold 1 --dataset_dict_path "$features_dir/all_dict.pickle" --features_deep_path "$features_dir/deep_features.pth" --features_path "$features_dir/concept_prior.csv" --save_path "$experiment_dir" --split_path "$split_path" --gene_expression_path "$gene_expression_path" 

python train_gecko_with_gene.py --keep_ratio 0.7 --top_k 10 --cross_val_fold 2 --dataset_dict_path "$features_dir/all_dict.pickle" --features_deep_path "$features_dir/deep_features.pth" --features_path "$features_dir/concept_prior.csv" --save_path "$experiment_dir" --split_path "$split_path" --gene_expression_path "$gene_expression_path" 

python train_gecko_with_gene.py --keep_ratio 0.7 --top_k 10 --cross_val_fold 3 --dataset_dict_path "$features_dir/all_dict.pickle" --features_deep_path "$features_dir/deep_features.pth" --features_path "$features_dir/concept_prior.csv" --save_path "$experiment_dir" --split_path "$split_path" --gene_expression_path "$gene_expression_path" 

python train_gecko_with_gene.py --keep_ratio 0.7 --top_k 10 --cross_val_fold 4 --dataset_dict_path "$features_dir/all_dict.pickle" --features_deep_path "$features_dir/deep_features.pth" --features_path "$features_dir/concept_prior.csv" --save_path "$experiment_dir" --split_path "$split_path" --gene_expression_path "$gene_expression_path" 


