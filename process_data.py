import numpy as np
import pandas as pd
import pickle
from reader import text_tokenizer
from transformers import AutoTokenizer
from os.path import join
import ast
import os
from scipy.stats import pearsonr, spearmanr


def get_score_vector_positions(args):
    return {trait:idx for idx, trait in enumerate(args.traits_to_use)}

def get_min_max_scores():
    return {
        1: {'overall_resolved': (2, 12), 'overall_strict': (1, 6), 'overall_lenient': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'overall_resolved': (1, 6), 'overall_strict': (1, 6), 'overall_lenient': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'overall_resolved': (0, 3), 'overall_strict': (0, 3), 'overall_lenient': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'overall_resolved': (0, 3), 'overall_strict': (0, 3), 'overall_lenient': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'overall_resolved': (0, 4), 'overall_strict': (0, 4), 'overall_lenient': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'overall_resolved': (0, 4), 'overall_strict': (0, 4), 'overall_lenient': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'overall_resolved': (0, 30), 'overall_strict': (0, 15), 'overall_lenient': (0, 15), 'content': (0, 6), 'organization': (0, 6), 'style': (0, 6), 'conventions': (0, 6)},
        8: {'overall_resolved': (0, 60), 'overall_strict': (0, 30), 'overall_lenient': (0, 30), 'content': (2, 12), 'organization': (2, 12), 'voice': (2, 12), 'word_choice': (2, 12),
            'sentence_fluency': (2, 12), 'conventions': (2, 12)}}


def deal_asap_pp(asap_pp_path):
    essays = {}
    for prompt_idx in range(1, 7):
        if prompt_idx in [1, 2]:
            attribute_score_indices = {
                'content': 1,
                'organization': 2,
                'word_choice': 3,
                'sentence_fluency': 4,
                'conventions': 5
            }
        else:
            attribute_score_indices = {
                'content': 1,
                'prompt_adherence': 2,
                'language': 3,
                'narrativity': 4
            }
        df_asap_pp = pd.read_csv(join(asap_pp_path, f'Prompt-{prompt_idx}.csv'))

        for idx, row in df_asap_pp.iterrows():
            essay_id = int(row['Essay ID'])
            essay = {'essay_id': essay_id}
            for trait, col_idx in attribute_score_indices.items():
                essay[trait] = int(row.iloc[col_idx])

            essays[essay_id] = essay

    return essays


def deal_asap(asap_path, tokenizer):
    essays = {}
    df_asap = pd.read_csv(asap_path, encoding='latin1')
    cnt = 0
    for idx, row in df_asap.iterrows():
        prompt_id = int(row['essay_set'])
        essay_id = int(row['essay_id'])
        essay = {'essay_id': essay_id, 'prompt_id': prompt_id, 'essay': row['essay'], 'overall_resolved': row['domain1_score'], 'overall_strict': row['rater1_domain1'], 'overall_lenient': row['rater2_domain1']}

        sent_tokens = text_tokenizer(row['essay'], replace_url_flag=True, tokenize_sent_flag=True)
        essay['essay_after_preprocessing'] = sent_tokens
        essay['encoded_essay'] = tokenizer(sent_tokens, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

        if essay['overall_strict'] > essay['overall_lenient']:
            essay['overall_strict'], essay['overall_lenient'] = essay['overall_lenient'], essay['overall_strict']

        if prompt_id in [7, 8]:
            if prompt_id == 7:
                attribute_score_indices = {
                    'content': (10, 16),
                    'organization': (11, 17),
                    'style': (12, 18),
                    'conventions': (13, 19)
                }
            elif prompt_id == 8:
                attribute_score_indices = {
                    'content': (10, 16, 22),
                    'organization': (11, 17, 23),
                    'voice': (12, 18, 24),
                    'word_choice': (13, 19, 25),
                    'sentence_fluency': (14, 20, 26),
                    'conventions': (15, 21, 27)
                }
            
            for trait, col_idx in attribute_score_indices.items():
                scores = [_ for _ in row.iloc[list(col_idx)].tolist() if not np.isnan(_)]
                if len(scores) == 2:
                    resolved_trait_score = int(sum(scores))
                elif len(scores) == 3:
                    resolved_trait_score = int(scores[-1] * 2)
                    cnt += 1
                else:
                    raise ValueError('Unexpected number of scores')

                essay[trait] = resolved_trait_score

        essays[essay_id] = essay

    return essays


def read_essays(args, tokenizer):

    asap_essays = deal_asap(args.asap_path, tokenizer)
    asap_pp_essays = deal_asap_pp(args.asap_pp_path)

    score_vector_positions = get_score_vector_positions(args)
    min_max_scores = get_min_max_scores()

    # merge asap_pp_essays and asap_essays based on keys
    # assert list(asap_pp_essays.keys()) == list(asap_essays.keys())
    for key in list(asap_essays.keys()):
        prompt_id = asap_essays[key]['prompt_id']
        if prompt_id < 7:
            if key not in asap_pp_essays:
                del asap_essays[key]
                continue
            asap_essays[key].update(asap_pp_essays[key])
        labels = [-1] * len(score_vector_positions)
        for trait in score_vector_positions:
            if trait not in asap_essays[key]:
                asap_essays[key][trait] = None
        for trait in min_max_scores[prompt_id].keys():
            if trait not in score_vector_positions: continue
            score = asap_essays[key][trait]
            scaled_score = (score - min_max_scores[prompt_id][trait][0]) / (min_max_scores[prompt_id][trait][1] - min_max_scores[prompt_id][trait][0])
            assert 0 <= scaled_score <= 1
            labels[score_vector_positions[trait]] = scaled_score
        asap_essays[key]['labels'] = tuple(labels)
    return asap_essays


def read_prompts(args):
    df_prompts = pd.read_csv(args.asap_prompts_path)
    df_prompts.sort_values('prompt_id', inplace=True)
    prompts = []
    for idx, row in df_prompts.iterrows():
        prompt = str(row['prompt'])
        prompts.append(prompt)
    return prompts

def ensure_list(x):
    if type(x) != list:
        return ast.literal_eval(x)
    return x

def get_all_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
    asap_prompts = read_prompts(args)
    encoded_prompts = tokenizer(asap_prompts, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

    asap_essays = read_essays(args, tokenizer)

    feature_groups = set()

    # load ridley's features if enabled
    if args.use_ridleys_feats:
        args.use_essay_feats = True

        # readability feats
        with open(args.ridleys_readability_path, 'rb') as f:
            readability_feats = pickle.load(f) # np array, [essay_id, feat_1, ..., feat_n]
        # iterate the array
        for i in range(readability_feats.shape[0]):
            essay_id = int(readability_feats[i][0])
            if essay_id not in asap_essays:
                continue
            feats = readability_feats[i][1:]
            asap_essays[essay_id]['readability_feats'] = feats.tolist()
        # hand crafted feats
        with open(args.ridleys_hand_crafted_path, 'rb') as f:
            hand_crafted_feats = pd.read_csv(f)
            hand_crafted_feats.set_index('essay_id', inplace=True)
            # iterate the dataframe
            for essay_id, row in hand_crafted_feats.iterrows():
                if essay_id not in asap_essays:
                    continue
                asap_essays[essay_id]['handcrafted_feats'] = row.values[1:].tolist()
        feature_groups.add('readability_feats')
        feature_groups.add('handcrafted_feats')

    if args.use_utos_feat or args.use_pronoun_feat or args.use_pos_features or args.use_sim_feat or args.use_top_n_words_feat or args.use_at_tokens_feat:
        args.use_essay_feats = True

        hand_crafted_feats_data = pd.read_csv(args.hand_crafted_feats_path)
        hand_crafted_feats_data.set_index('essay_id', inplace=True)
        # iterate the dataframe
        for essay_id, row in hand_crafted_feats_data.iterrows():
            if essay_id not in asap_essays:
                continue
            if args.use_utos_feat:
                asap_essays[essay_id]['utos_feat'] = ensure_list(row['utos_feat'])
                feature_groups.add('utos_feat')
            if args.use_pronoun_feat:
                asap_essays[essay_id]['pronoun_feat'] = ensure_list(row['pronoun_feat'])
                feature_groups.add('pronoun_feat')
            if args.use_pos_features:
                asap_essays[essay_id]['pos_feat'] = ensure_list(row['pos_feat'])
                feature_groups.add('pos_feat')
            if args.use_sim_feat:
                asap_essays[essay_id]['sent_sim_feat'] = ensure_list(row['sent_sim_feat'])
                feature_groups.add('sent_sim_feat')
            if args.use_top_n_words_feat:
                asap_essays[essay_id]['top_n_words_feat'] = ensure_list(row['top_n_words_feat'])
                feature_groups.add('top_n_words_feat')
            if args.use_at_tokens_feat:
                asap_essays[essay_id]['at_tokens_feat'] = ensure_list(row['at_tokens_feat'])
                feature_groups.add('at_tokens_feat')

    # drop essay 4355 from df_asap because it does not appear in ridley's folds
    if 4355 in asap_essays:
        del asap_essays[4355]

    return asap_essays, asap_prompts, encoded_prompts, feature_groups


def partition_data(args, asap_essays):

    with open(args.folds_path + f'fold_{args.test_fold_idx}/train_ids.txt', 'r') as f:
        train_ids = [int(line.strip()) for line in f]
    with open(args.folds_path + f'fold_{args.test_fold_idx}/dev_ids.txt', 'r') as f:
        dev_ids = [int(line.strip()) for line in f]
    with open(args.folds_path + f'fold_{args.test_fold_idx}/test_ids.txt', 'r') as f:
        test_ids = [int(line.strip()) for line in f]

    train_data = [asap_essays[essay_id] for essay_id in train_ids if essay_id in asap_essays]
    dev_data = [asap_essays[essay_id] for essay_id in dev_ids if essay_id in asap_essays]
    test_data = [asap_essays[essay_id] for essay_id in test_ids if essay_id in asap_essays]

    return train_data, dev_data, test_data


def get_partitions(args):
    if args.cache_path and os.path.exists(args.cache_path):
        with open(args.cache_path, 'rb') as f:
            train_data, dev_data, test_data, asap_essays, asap_prompts, encoded_prompts = pickle.load(f)

        feature_groups = set()
        if args.use_ridleys_feats:
            feature_groups.add('readability_feats')
            feature_groups.add('handcrafted_feats')
        if args.use_utos_feat:
            feature_groups.add('utos_feat')
        if args.use_pronoun_feat:
            feature_groups.add('pronoun_feat')
        if args.use_pos_features:
            feature_groups.add('pos_feat')
        if args.use_sim_feat:
            feature_groups.add('sent_sim_feat')
        if args.use_top_n_words_feat:
            feature_groups.add('top_n_words_feat')
        if args.use_at_tokens_feat:
            feature_groups.add('at_tokens_feat')
        if feature_groups:
            args.use_essay_feats = True
    else:
        asap_essays, asap_prompts, encoded_prompts, feature_groups = get_all_data(args)

        train_data, dev_data, test_data = partition_data(args, asap_essays)

        if args.cache_path:
            with open(args.cache_path, 'wb') as f:
                pickle.dump((train_data, dev_data, test_data, asap_essays, asap_prompts, encoded_prompts), f)

    # Feature selection
    if args.use_essay_feats:
        if float(args.threshold) > 0:
            feature_group_to_needed_feats = feature_selection(args, train_data, feature_groups)
            # update asap_essays with the selected features
            for essay_id, essay in asap_essays.items():
                asap_essays[essay_id]['essay_feats'] = []
                for feature_group in feature_group_to_needed_feats:
                    indices = feature_group_to_needed_feats[feature_group]
                    filtered_feats = [essay[feature_group][idx] for idx in indices]
                    asap_essays[essay_id]['essay_feats'].extend(filtered_feats)
            # re-partition the data
            train_data, dev_data, test_data = partition_data(args, asap_essays)
        else:
            # calculate essay_feats_size
            args.essay_feats_size = 0
            for feature_group in feature_groups:
                args.essay_feats_size += len(train_data[0][feature_group])
            # update asap_essays with the selected features
            for essay_id, essay in asap_essays.items():
                asap_essays[essay_id]['essay_feats'] = []
                for feature_group in feature_groups:
                    asap_essays[essay_id]['essay_feats'].extend(essay[feature_group])
    return train_data, dev_data, test_data, asap_essays, asap_prompts, encoded_prompts, feature_groups

def feature_selection(args, train_essays, feature_groups):
    prompt_ids = np.array([essay['prompt_id'] for essay in train_essays])
    y = np.array([essay['labels'][0] for essay in train_essays])
    args.essay_feats_size = 0
    feature_group_to_needed_feats = {}
    for feature_group in feature_groups:
        feats = np.array([essay[feature_group] for essay in train_essays])
        needed_feats = []
        print(f'Feature group: {feature_group}, size before selection: {feats.shape}')
        setattr(args, f'size_before_selection_{feature_group}', feats.shape)
        if len(feats) == 0:
            continue
        # calculate the min of pearson correlation and spearman correlation between each feature and y
        for i in range(feats.shape[1]):
            corr_list = []
            for prompt_id in np.unique(prompt_ids):
                mask = prompt_ids == prompt_id
                _x = feats[mask, i]
                _y = y[mask]
                _p, _ = pearsonr(_x, _y)
                _sp, _ = spearmanr(_x, _y)
                corr = min(abs(_p), abs(_sp))
                corr_list.append(corr)
            if np.nanmean(corr_list) >= float(args.threshold):
                needed_feats.append(i)
        print(f'Feature group: {feature_group}, size after selection: {len(needed_feats)}')
        setattr(args, f'size_afterselection_{feature_group}', len(needed_feats))
        feature_group_to_needed_feats[feature_group] = needed_feats
        args.essay_feats_size += len(needed_feats)
    return feature_group_to_needed_feats


def infer_trait_weights_for_loss(args, train_data):
    trait_counts = {trait:0 for trait in args.traits_to_use}
    for essay in train_data:
        for trait, label in zip(args.traits_to_use, essay['labels']):
            if label != -1:
                trait_counts[trait] += 1
    for trait, trait_count in trait_counts.items():
        trait_counts[trait] = 1 / trait_counts[trait]
    total_weight = sum(trait_counts.values())
    for trait in trait_counts:
        trait_counts[trait] /= total_weight
        trait_counts[trait] *= len(trait_counts)
    trait_weights = [trait_counts[trait] for trait in args.traits_to_use]
    return trait_weights


def remove_training_instance(data, traits_to_use):
    new_data = []
    for essay in data:
        if any([trait != -1 for trait in essay['labels']]):
            new_data.append(essay)
    return new_data