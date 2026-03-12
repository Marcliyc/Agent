from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.agent.executor import SafeExecutor
from src.features.clean import apply_cleaning, build_cleaning_plan
from src.features.diagnose import run_feature_diagnosis
from src.features.evaluate import add_redundancy_penalty, compute_final_scores
from src.features.select import select_top_features
from src.modeling.validate import train_and_compare
from src.utils.config import PipelineConfig
from src.utils.data_io import chronological_split, detect_leakage, ensure_columns_exist, load_dataset


def label_screening(train_df, label_cols):
    candidates = []
    for y in label_cols:
        if y not in train_df.columns:
            continue
        s = train_df[y]
        miss = float(s.isna().mean())
        values = s.dropna().unique()
        if len(values) != 2:
            continue
        balance = float(min((s == values[0]).mean(), (s == values[1]).mean()))
        score = (1 - miss) * balance
        candidates.append((y, score, miss, balance))

    if not candidates:
        raise RuntimeError('No suitable binary label found in Y1~Y12')
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0], candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/data.pq')
    parser.add_argument('--out-dir', type=str, default='Q2/outputs')
    args = parser.parse_args()

    cfg = PipelineConfig(data_path=Path(args.data_path), output_dir=Path(args.out_dir))
    cfg.ensure_dirs()
    random.seed(cfg.random_seed)

    ex = SafeExecutor()

    df = ex.run('load_dataset', load_dataset, cfg.data_path)
    split = ex.run('chronological_split', chronological_split, df, 'trade_date')
    leakage = ex.run('leakage_check', detect_leakage, split, 'trade_date')

    train_df, valid_df, test_df = split['train'], split['valid'], split['test']
    target_col, label_rank = ex.run('label_screening', label_screening, train_df, cfg.label_columns)

    feature_cols = ensure_columns_exist(train_df, cfg.feature_columns)

    diagnosis = ex.run('feature_diagnosis', run_feature_diagnosis, train_df, feature_cols, target_col)
    cleaning = ex.run('build_cleaning_plan', build_cleaning_plan, diagnosis)
    clean_train, clean_valid, clean_test = ex.run(
        'apply_cleaning', apply_cleaning, train_df, valid_df, test_df, cleaning, target_col
    )

    usable_features = [c for c in feature_cols if c in clean_train.columns]
    redundancy = ex.run('redundancy', add_redundancy_penalty, clean_train, usable_features, cfg.corr_threshold)
    scored = ex.run('score_features', compute_final_scores, diagnosis[diagnosis['feature'].isin(usable_features)], redundancy)
    top50 = ex.run('select_top50', select_top_features, scored, cfg.top_k, cfg.corr_threshold, clean_train)

    random50 = random.sample(usable_features, k=min(cfg.top_k, len(usable_features)))
    feature_sets = {
        'raw_all': feature_cols,
        'cleaned_all': usable_features,
        'agent_top50': top50['feature'].tolist(),
        'random_50': random50,
    }
    metrics = ex.run(
        'model_validation',
        train_and_compare,
        clean_train,
        clean_valid,
        clean_test,
        target_col,
        feature_sets,
        cfg.random_seed,
    )

    diagnosis.to_csv(cfg.output_dir / 'feature_diagnosis.csv', index=False)
    cleaning.to_csv(cfg.output_dir / 'feature_cleaning_log.csv', index=False)
    scored.to_csv(cfg.output_dir / 'feature_scores.csv', index=False)
    top50.to_csv(cfg.output_dir / 'selected_top50.csv', index=False)
    metrics.to_csv(cfg.output_dir / 'model_metrics.csv', index=False)

    with open(cfg.output_dir / 'agent_execution_log.json', 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in ex.records], f, ensure_ascii=False, indent=2)

    with open(cfg.output_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump({'target_label': target_col, 'label_screening': label_rank, 'leakage': leakage}, f, ensure_ascii=False, indent=2)

    print('Pipeline completed')
    print('target_label=', target_col)
    print('leakage=', leakage)


if __name__ == '__main__':
    main()
