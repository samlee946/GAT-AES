import process_data
import argparse
import torch
import numpy as np
import tqdm
import wandb
import logging
import random
import os
import pickle
import torch_geometric


from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from GNN import GNN, MultiNodeGNN, MultiNodeGNNForNVEmbed, GNNForNVEmbed, MultiNodeGCN, MultiNodeGNNWithPromptNodes
import TraitAttention
from BertBaseLine import BL
from metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--lm_lr', type=float, default=3e-5, help='lr for lm')
    # parser.add_argument('--warmup_ratio', type=float, help='warmup ratio', default=0.1)
    # parser.add_argument('--lds_sigma', type=int, default=2)
    # parser.add_argument('--lds_ks', type=float, default=15)
    # parser.add_argument('--disable_lora', action='store_true', help='disable lora')
    # parser.add_argument('--disable_lm', action='store_true', help='disable lm')
    # parser.add_argument('--aux_task_prompt_classification_ce', action='store_true', help='')
    # parser.add_argument('--aux_task_prompt_classification_bce', action='store_true', help='')
    # parser.add_argument('--aux_task_prompt_classification_weight', type=float, default=1)
    # parser.add_argument('--log_dir', type=str, help='log directory', default='logs/')
    # # accepts a list of hidden sizes
    # parser.add_argument('--act_func', type=str, default='relu')
    # parser.add_argument('--train_dev_partition_name', type=str, help='train dev partition name', default='asap_ridley')
    # parser.add_argument('--enable_flash_attention', action='store_true', help='enable flash attention')
    

    # Path related
    parser.add_argument('--cache_path', type=str, help='path to store preprocessed dataset', default=None)
    parser.add_argument('--asap_path', type=str, help='path to asap-aes dataset', default='data/training_set_rel3_latin1_with_all_feats.csv')
    parser.add_argument('--asap_prompts_path', type=str, help='path to asap-aes prompts', default='data/asap_prompts_with_source.csv')
    parser.add_argument('--asap_pp_path', type=str, help='path to asap plus plus', default='data/ASAP++/')
    parser.add_argument('--ridleys_readability_path', type=str, help='path to the rubrics', default='data/allreadability_updated.pickle')
    parser.add_argument('--ridleys_hand_crafted_path', type=str, help='path to the rubrics', default='data/hand_crafted_v3_normalized.csv')
    parser.add_argument('--lis_feats_path', type=str, help='path to the rubrics', default='data/training_set_rel3_latin1_with_all_feats.csv')
    parser.add_argument('--llm_feats_path', type=str, help='path to the rubrics', default='data/preprocessed_dataset_with_llm_merged_standardized.pkl')
    parser.add_argument('--folds_path', type=str, help='path to the folds', default='data/Taghipour_folds/')
    parser.add_argument('--output_path', type=str, help='path to store output', default='output/')


    # Features related
    parser.add_argument('--threshold', type=str, help='activation function', default=-1)
    parser.add_argument('--use_ridleys_feats', action='store_true', help='use ridleys features')    
    parser.add_argument('--use_llm_features_llama', action='store_true', default=False)
    parser.add_argument('--use_llm_features_gemma', action='store_true', default=False)
    parser.add_argument('--use_utos_feat', action='store_true', help='use pos features')
    parser.add_argument('--use_pos_features', action='store_true', help='use pos features')
    parser.add_argument('--use_top_n_words_feat', action='store_true', help='use sentence similarity features')
    parser.add_argument('--use_pronoun_feat', action='store_true', help='use pronoun features')
    parser.add_argument('--use_sim_feat', action='store_true', help='use sentence similarity features')
    parser.add_argument('--use_at_tokens_feat', action='store_true', help='use features related to at tokens')
    parser.add_argument('--use_essay_feats', action='store_true', help='use essay features')

    # Logging related
    parser.add_argument('--wandb_project_name', type=str, help='wandb project name', default='AES-within-prompt')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--disable_model_checkpoint', action='store_true', help='use essay features')

    # Hyperparameters
    parser.add_argument('--lm_model', type=str, help='pretrained language model', default='google-bert/bert-base-cased')
    parser.add_argument('--task_lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=3)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--seed', type=int, help='random seed', default=11)
    parser.add_argument('--num_heads', type=int, help='number of heads', default=1)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', help='hidden sizes', default=[])
    parser.add_argument('--traits_to_use', type=str, nargs='+', help='traits to use', default=['overall_resolved', 'overall_strict', 'overall_lenient', 'content', 'organization', 'word_choice', 'sentence_fluency', 'conventions', 'prompt_adherence', 'language', 'narrativity'])
    parser.add_argument('--test_fold_idx', type=int, required=True, help='fold id to test')
    parser.add_argument('--layer_weight_scheme', type=str, help='')
    parser.add_argument('--dropout', type=float, help='dropout rate', default=-1)
    parser.add_argument('--fixed_weight_for_traits', action='store_true', help='use fixed weights for traits')
    parser.add_argument('--trait_weights_for_loss', type=float, nargs='+', help='weights for traits in loss', default=None)
    parser.add_argument('--infer_trait_weights_for_loss', type=str, help='infer trait weights for loss', default='none', choices=['learnable', 'fixed', 'none'])
    parser.add_argument('--essay_feats_size', type=int, help='size of essay features', default=0)
    parser.add_argument('--loss_reduction', type=str, help='', default='sum', choices=['sum', 'mean'])
    parser.add_argument('--dim_reduction', type=int, default=-1)
    parser.add_argument('--apply_max_pooling', action='store_true', help='')
    parser.add_argument('--freeze_layers_num', type=int, default=-1)
    parser.add_argument('--use_trait_att', action='store_true', help='')
    parser.add_argument('--use_prompt_nodes', action='store_true', help='')
    parser.add_argument('--run_bert_large_base_line', action='store_true', help='')
    parser.add_argument('--run_t5_base_line', action='store_true', help='')
    

    # GAT Hyperparameters
    parser.add_argument('--GNN_hidden_size', type=int, default=64)
    parser.add_argument('--GNN_num_heads', type=int, default=4)
    parser.add_argument('--GNN_num_layers', type=int, default=2)
    parser.add_argument('--GNN_num_emb_nodes', type=int, default=1)
    parser.add_argument('--GNN_num_feat_nodes', type=int, default=1)
    parser.add_argument('--GNN_num_prompt_nodes', type=int, default=1)
    parser.add_argument('--GNN_dropout', type=float, default=0.1)
    parser.add_argument('--GNN_edge_mode', type=str, default='all_pairs', choices=['all_pairs', 'none', 'arts_order', 'highly_correlated'])
    parser.add_argument('--GNN_edge_correlation_threshold', type=float, default=0)
    parser.add_argument('--GNN_layer_type', type=str, default='GAT', choices=['GAT', 'GCN', 'SAGE', 'GIN'])


    args = parser.parse_args()

    # init random seed
    set_seed(args.seed)

    # init wandb logging
    wandb.init(
        project = args.wandb_project_name,
        name = args.run_name,
    )

    wandb.define_metric("train_log/loss", summary="min")
    wandb.define_metric("dev_log/loss", summary="min")
    wandb.define_metric("test_log/loss", summary="min")
    for trait in args.traits_to_use:
        for prompt_id in range(1, 9):
            wandb.define_metric(f"dev_log/qwk_t@{prompt_id}_{trait}", summary="max")
            wandb.define_metric(f"test_log/qwk_t@{prompt_id}_{trait}", summary="max")

    # init logger
    logger = logging.getLogger('AES')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if (logger.hasHandlers()):
        logger.handlers.clear()
    ## stream handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    ## file handler
    os.makedirs(args.output_path, exist_ok=True)
    handler = logging.FileHandler(os.path.join(args.output_path, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # load data
    train_data, dev_data, test_data, asap_essays, asap_prompts, encoded_prompts, feature_groups = process_data.get_partitions(args)

    # infer_trait_weights_for_loss
    if args.infer_trait_weights_for_loss != 'none':
        args.trait_weights_for_loss = process_data.infer_trait_weights_for_loss(args, train_data)

    # if not all traits are used, remove the ones that do not have any labels
    if len(args.traits_to_use) < 11:
        train_data = process_data.remove_training_instance(train_data, args.traits_to_use)
        dev_data = process_data.remove_training_instance(dev_data, args.traits_to_use)
        test_data = process_data.remove_training_instance(test_data, args.traits_to_use)

    # create data loaders using torch
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    train_loader = create_data_loader(args, train_data, args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = create_data_loader(args, dev_data, args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = create_data_loader(args, test_data, args.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.run_name == 'debug':
        # sample a subset for debugging
        # train_loader = DataLoader(train_data[:100], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        # dev_loader = DataLoader(dev_data[:100], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        # test_loader = DataLoader(test_data[:100], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        train_loader = create_data_loader(args, train_data[:100], args.batch_size, shuffle=True, collate_fn=collate_fn)
        dev_loader = create_data_loader(args, dev_data[:100], args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = create_data_loader(args, test_data[:100], args.batch_size, shuffle=True, collate_fn=collate_fn)

    # define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_bert_large_base_line:
        model = BL(args, encoded_prompts)
    elif args.use_trait_att:
        model = TraitAttention.MultiNodeGNN(args, encoded_prompts)
    elif args.use_prompt_nodes:
        model = MultiNodeGNNWithPromptNodes(args, encoded_prompts)
    elif args.GNN_num_feat_nodes > 1 or args.GNN_num_emb_nodes > 1 or args.GNN_layer_type != 'GAT':
        if 'nvidia' in args.lm_model:
            model = MultiNodeGNNForNVEmbed(args, encoded_prompts)
        else:
            if args.GNN_layer_type == 'GAT':
                model = MultiNodeGNN(args, encoded_prompts)
            else:
                model = MultiNodeGCN(args, encoded_prompts)
    else:
        if 'nvidia' in args.lm_model:
            model = GNNForNVEmbed(args, encoded_prompts)
        else:
            model = GNN(args, encoded_prompts)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.task_lr)  # AdamW

    # log things
    logger.info(f"Train data size: {len(train_data)}")
    logger.info(f"Dev data size: {len(dev_data)}")
    logger.info(f"Test data size: {len(test_data)}")
    logger.info(f"args: \n{args}")
    logger.info(f"Model: \n{model}")

    best_dev_metrics = None
    best_test_metrics = None
    best_epoch = -1
    max_qwk_avg = -1

    # init random seed
    set_seed(args.seed)

    # train the model
    with tqdm.tqdm(total=args.epochs * len(train_loader), desc="Training", unit="batch") as pbar:
        for epoch in range(args.epochs):
            
            wandb_log_dict = {}

            for train_batch in train_loader:

                train_batch = to_cuda(train_batch, device)

                model.train()
                optimizer.zero_grad()
                loss, preds = model(**train_batch)

                wandb.log({
                    "train_log/loss": loss.item(),
                    "train_log/step": pbar.n,
                    "train_log/fold_idx": args.test_fold_idx
                })

                loss.backward()
                optimizer.step()

                pbar.update(1)

            model.eval()
            with torch.no_grad():
                dev_metrics, dev_essay_ids, dev_prompt_ids, dev_preds, dev_labels, dev_losses = evaluate(args, model, dev_loader, device)
                test_metrics, test_essay_ids, test_prompt_ids, test_preds, test_labels, test_losses = evaluate(args, model, test_loader, device)

            if not best_dev_metrics or dev_metrics['qwk_avg'] > best_dev_metrics['qwk_avg']:
                best_dev_metrics = dev_metrics
                best_test_metrics = test_metrics
                best_epoch = epoch

                # every 10 epochs, log the best metrics
                epoch_10 = (epoch + 10) // 10 * 10
                wandb_log_dict[f"best_dev_qwk_avg_{epoch_10}"] = best_dev_metrics
                wandb_log_dict[f"best_test_qwk_avg_{epoch_10}"] = best_test_metrics

                # save best model 
                if not args.disable_model_checkpoint:
                    torch.save(model.state_dict(), os.path.join(args.output_path, "best_model.pth"))

            # save output to files
            with open(os.path.join(args.output_path, f"Output_{epoch}.pkl"), "wb") as f:
                pickle.dump({
                    "dev_metrics": dev_metrics,
                    "dev_essay_ids": dev_essay_ids,
                    "dev_prompt_ids": dev_prompt_ids,
                    "dev_preds": dev_preds,
                    "dev_labels": dev_labels,
                    "dev_losses": dev_losses,
                    "test_metrics": test_metrics,
                    "test_essay_ids": test_essay_ids,
                    "test_prompt_ids": test_prompt_ids,
                    "test_preds": test_preds,
                    "test_labels": test_labels,
                    "test_losses": test_losses,
                }, f)

            wandb_log_dict["dev_log/epoch"] = epoch
            wandb_log_dict["dev_log/loss"] = np.mean(dev_losses)
            wandb_log_dict["test_log/loss"] = np.mean(test_losses)
            for k, v in dev_metrics.items():
                wandb_log_dict[f"dev_log/{k}"] = v
            for k, v in test_metrics.items():
                wandb_log_dict[f"test_log/{k}"] = v
            wandb_log_dict["best_epoch"] = best_epoch
            wandb_log_dict["best_dev_qwk_avg"] = best_dev_metrics['qwk_avg']
            wandb_log_dict["best_test_qwk_avg"] = best_test_metrics['qwk_avg']
            max_qwk_avg = max(max_qwk_avg, best_test_metrics['qwk_avg'])
            wandb_log_dict["max_test_qwk_avg"] = max_qwk_avg
            wandb.log(wandb_log_dict)
            
            logger.info(f"""dev metrics: {dev_metrics}""")
            logger.info(f"""test metrics: {test_metrics}""")

            logger.info(f"""
Epoch:         {epoch}
Dev QWK:       {dev_metrics['qwk_avg']:.4f}
Test QWK:      {test_metrics['qwk_avg']:.4f}
Best Epoch:    {best_epoch}
Best Dev QWK:  {best_dev_metrics['qwk_avg']:.4f}
Best Test QWK: {best_test_metrics['qwk_avg']:.4f}
""")
            pbar.set_description(f"Epoch: {epoch}/{args.epochs}, Best Dev QWK: {best_dev_metrics['qwk_avg']:.4f}, Best Test QWK: {best_test_metrics['qwk_avg']:.4f}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader(args, dataset, batch_size, shuffle, collate_fn):
    # Create a generator that can be seeded
    g = torch.Generator()
    g.manual_seed(args.seed)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        generator=g,
        worker_init_fn=seed_worker  # We'll define this function
    )


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if set_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These settings are needed for CUDA >= 10.2
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Add these PyTorch Geometric specific settings
    if hasattr(torch_geometric, 'seed_everything'):
        torch_geometric.seed_everything(seed)
    
    # Force deterministic behavior for GATConv
    os.environ['TORCH_DETERMINISTIC'] = '1'

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed) # this does not work for me. I suggest setting PYTHONHASHSEED when running the script


def evaluate(args, model, data_loader, device):
    essay_ids = []
    prompt_ids = []
    preds = []
    labels = []
    losses = []
    for batch in data_loader:
        essay_ids.append(batch['essay_ids'].cpu().numpy())
        prompt_ids.append(batch['prompt_ids'].cpu().numpy())
        labels.append(batch['labels'].cpu().numpy())
        batch = to_cuda(batch, device)
        loss, batch_preds = model(**batch)
        preds.append(batch_preds.cpu().numpy())
        losses.append(loss.item())

    essay_ids = np.concatenate(essay_ids)
    prompt_ids = np.concatenate(prompt_ids)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    losses = np.array(losses)

    metrics = compute_metrics(args, prompt_ids, preds, labels)
    # import ipdb; ipdb.set_trace()
    return metrics, essay_ids, prompt_ids, preds, labels, losses

def to_cuda(batch, device):
    return {key: value.to(device) if value is not None else None for key, value in batch.items()}


def collate_fn(batch):
    essay_input_ids = torch.stack([item["encoded_essay"]["input_ids"] for item in batch])
    essay_attention_mask = torch.stack([item["encoded_essay"]["attention_mask"] for item in batch])
    if "token_type_ids" in batch[0]["encoded_essay"]:
        essay_token_type_ids = torch.stack([item["encoded_essay"]["token_type_ids"] for item in batch])
    else:
        essay_token_type_ids = None
    # encoded_essays = ([item["encoded_essay"] for item in batch])
    essay_ids = torch.tensor([item["essay_id"] for item in batch])
    prompt_ids = torch.tensor([item["prompt_id"] for item in batch])
    essay_feats = torch.tensor([item["essay_feats"] for item in batch]) if "essay_feats" in batch[0] else None
    labels = torch.tensor([item["labels"] for item in batch])
    # essay_after_preprocessing = None#[item["essay_after_preprocessing"] for item in batch]
    # return essay_ids, prompt_ids, essay_input_ids, essay_attention_mask, essay_token_type_ids, essay_feats, labels
    return {
        "essay_input_ids": essay_input_ids,
        "essay_attention_mask": essay_attention_mask,
        "essay_token_type_ids": essay_token_type_ids,
        # "encoded_essays": encoded_essays,
        "essay_ids": essay_ids,
        "prompt_ids": prompt_ids,
        "essay_feats": essay_feats,
        "labels": labels
    }


if __name__ == '__main__':
    main()