import process_data
from sklearn.metrics import cohen_kappa_score, mean_squared_error as mse, mean_absolute_error as mae, r2_score, mean_squared_error as mse, root_mean_squared_error as rmse
from scipy.stats import pearsonr as pear
import numpy as np
from collections import defaultdict


def compute_metrics(args, prompt_ids, preds, labels):
    # rescale preds and labels
    min_max_scores = process_data.get_min_max_scores()
    score_vector_positions = process_data.get_score_vector_positions(args)
    metrics = {}
    prompt_scores = defaultdict(list)
    for trait, trait_idx in score_vector_positions.items():
        qwk_scores = []
        pear_scores = []
        mse_scores = []
        rmse_scores = []
        r2_scores = []
        mae_scores = []
        for prompt_id in np.unique(prompt_ids):
            if trait not in min_max_scores[prompt_id]:
                continue
            low, high = min_max_scores[prompt_id][trait]
            mask = prompt_ids == prompt_id
            preds[mask, trait_idx] = preds[mask, trait_idx] * (high - low) + low
            labels[mask, trait_idx] = labels[mask, trait_idx] * (high - low) + low

            # round to nearest integer
            trait_preds = np.round(preds[mask, trait_idx]).astype(int)
            trait_labels = np.round(labels[mask, trait_idx]).astype(int)

            _qwk_score = cohen_kappa_score(trait_preds, trait_labels, weights='quadratic')
            _pear_score = pear(trait_preds, trait_labels)[0]
            _mse_score = mse(trait_preds, trait_labels).item()
            _rmse_score = rmse(trait_preds, trait_labels).item()
            _r2_score = r2_score(trait_preds, trait_labels)
            _mae_score = mae(trait_preds, trait_labels).item()

            metrics[f'qwk_t@{prompt_id}_{trait}']       = _qwk_score
            metrics[f'pear_t@{prompt_id}_{trait}']      = _pear_score
            metrics[f'mse_t@{prompt_id}_{trait}']       = _mse_score
            metrics[f'rmse_t@{prompt_id}_{trait}']      = _rmse_score
            metrics[f'r2_score_t@{prompt_id}_{trait}']  = _r2_score
            metrics[f'mae_t@{prompt_id}_{trait}']       = _mae_score

            prompt_scores[prompt_id].append(_qwk_score)

            qwk_scores.append(_qwk_score)
            pear_scores.append(_pear_score)
            mse_scores.append(_mse_score)
            rmse_scores.append(_rmse_score)
            r2_scores.append(_r2_score)
            mae_scores.append(_mae_score)

        metrics[f'qwk_{trait}_avg'] = np.nan if len(qwk_scores) == 0 else np.nanmean(qwk_scores)
        metrics[f'pear_{trait}_avg'] = np.nan if len(pear_scores) == 0 else np.nanmean(pear_scores)
        metrics[f'mse_{trait}_avg'] = np.nan if len(mse_scores) == 0 else np.nanmean(mse_scores)
        metrics[f'rmse_{trait}_avg'] = np.nan if len(rmse_scores) == 0 else np.nanmean(rmse_scores)
        metrics[f'r2_score_{trait}_avg'] = np.nan if len(r2_scores) == 0 else np.nanmean(r2_scores)
        metrics[f'mae_{trait}_avg'] = np.nan if len(mae_scores) == 0 else np.nanmean(mae_scores)

    for prompt_id in range(1, 9):
        metrics[f'qwk_avg@{prompt_id}'] = np.nanmean(prompt_scores[prompt_id])

    metrics['qwk_avg'] = np.nanmean([metrics[f'qwk_{trait}_avg'] for trait in score_vector_positions.keys()])
    metrics['qwk_avg_prompts'] = np.nanmean([metrics[f'qwk_avg@{prompt_id}'] for prompt_id in range(1, 9)])
 
    return metrics