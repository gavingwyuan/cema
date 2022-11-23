# CEMA
This folder contains code to run models from our paper (CEMA – Cost-Efficient Machine-Assisted Document Annotations).

# Installation
conda create -n cema python=3.7.6  
conda activate cema  

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html  
pip install transformers==4.9.0  
pip install fitlog==0.9.13  
pip install nervaluate==0.1.8  
pip install nltk==3.6.2  
pip install pandas==1.3.0  
pip install python_crfsuite==0.9.8  
pip install scikit_learn  
pip install seqeval==1.2.2  
pip install hiddenlayer==0.3  
pip install matplotlib  
pip install jupyterlab  

# Data preparing
Data files are in the folder 'dataset'. The arguments (labels_map_file, train_file, test_file) for paths setting can be found in 'get_args.py'.

label2idx_dict: It is a dict save in json format, which map label to index.
train_file, val_file and test_file: They are csv files. Two columns 'token_seq' and 'label_seq' are required. Each cell at 'token_seq' stores a word list. The cell in the same row and at the column 'label_seq' stores a label list, which stores labels for each word in the word list.

# Run Experiments
CUDA_VISIBLE_DEVICES=0 python k_folds_demo_cema.py

Then, the results will be ouput to the file 'results/german_doc/maa_dn2_itv_3_init_ep50_5_bs8/cema_partial_normal_w0.5_tr_0.3_budget_30_sim_0.8/dbmdz/bert-base-german-cased_tagger_r_1_qs_cema_partial_normal.intermediate_results.pkl'.

Next, run the file 'data_processed/draw_cost_f1_figure.ipynb' to a draw cost-F1 figure for the results.

# Evaluation
Entity level F1 score is evaluated between predicted markups and correct markups. We provide a program to draw a figure of cost-F1 score and output cost-F1 to a csv-format. The program file is in 'data_processed/draw_cost_f1_figure.ipynb', and the output files are in 'data_processed/csv'.

# Hardware requirements
We have run experiments on GeForce RTX 2080 Ti with 11GB of memory.

