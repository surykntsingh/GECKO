# WSIs Feature Extraction Pipeline

This repository includes script to extract both Concept Prior and Deep features from Whole Slide Images (WSIs). It also contains several custom Python files for patch extraction, deep feature extraction using the CONCH model, and Concept-Prior extraction.


Run the following command for end-to-end data curation:
```bash
chmod +x curate.sh
./curate.sh /your/slide_dir /your/patch_dir /your/lists_feats_dir /path/to/conch_weight /your/patch_prompts
```

where /your/slide_dir: refers to directory of WSIs, 
/your/patch_dir: refers to empty directory where patches will be saved, 
/your/lists_feats_dir: refers to empty directory where patch list and features will be saved, 
/path/to/conch_weight: refers to path of 'pytorch_model.bin' from CONCH, 
and /your/patch_prompts: refers to patch_prompts.json for corresponding dataset in prompts/ directory.  

## Important Notes

- For patch extraction, magnification of 20X and size of 448x448 px is set as default in line with specifications from CONCH model. 
- For extending to new dataset, please make sure to have patch_prompts in similar format as provided for multiple datasets in this repo. 

Feel free to raise issues or contribute to this project if you have any improvements or encounter any problems.
