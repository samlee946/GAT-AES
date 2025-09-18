# Graph-Based Multi-Trait Essay Scoring (GAT-AES)

This repo stores code for the EMNLP 2025 paper: Graph-Based Multi-Trait Essay Scoring.

## Usage
```
python train_GNN.py --epochs 15 --GNN_hidden_size 512 --GNN_num_layers 2 --traits_to_use overall_resolved content organization word_choice sentence_fluency conventions prompt_adherence language narrativity style voice --use_ridleys_feats --use_utos_feat --use_pos_features --use_top_n_words_feat --use_pronoun_feat --use_sim_feat --use_essay_feats --threshold 0.2 --test_fold_idx 3 --task_lr 3e-5 --seed 1337 --lm_model google-bert/bert-large-cased --GNN_num_feat_nodes 2 --GNN_num_emb_nodes 1 --GNN_dropout 0.1 --batch_size 16
```

## Trained Checkpoint
https://drive.google.com/file/d/1SGNRq847vy--wNanPtlaAEO-E0VQmQFz/view?usp=drive_link

## Bibtex
WIP