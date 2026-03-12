from __future__ import annotations


def build_cleaning_plan(diagnosis_df):
    plans = []
    for _, row in diagnosis_df.iterrows():
        actions = []
        if row['missing_rate'] > 0.0:
            actions.append('median_impute_by_train')
        if row['outlier_rate'] > 0.05:
            actions.append('winsorize_1_99')
        if abs(row['skewness']) > 2.0:
            actions.append('signed_log1p')
        if row['missing_rate'] > 0.8 or row['unique_ratio'] < 1e-4:
            actions.append('drop_feature')
        plans.append({'feature': row['feature'], 'cleaning_action': ';'.join(actions) or 'none'})

    import pandas as pd

    return pd.DataFrame(plans)


def apply_cleaning(train_df, valid_df, test_df, cleaning_df, target_col: str):
    import numpy as np

    train, valid, test = train_df.copy(), valid_df.copy(), test_df.copy()

    for _, row in cleaning_df.iterrows():
        feat = row['feature']
        actions = row['cleaning_action'].split(';') if row['cleaning_action'] else []
        if 'drop_feature' in actions:
            for part in (train, valid, test):
                if feat in part.columns:
                    del part[feat]
            continue

        if 'median_impute_by_train' in actions and feat in train.columns:
            med = train[feat].median()
            train[feat] = train[feat].fillna(med)
            valid[feat] = valid[feat].fillna(med)
            test[feat] = test[feat].fillna(med)

        if 'winsorize_1_99' in actions and feat in train.columns:
            lo = train[feat].quantile(0.01)
            hi = train[feat].quantile(0.99)
            train[feat] = train[feat].clip(lo, hi)
            valid[feat] = valid[feat].clip(lo, hi)
            test[feat] = test[feat].clip(lo, hi)

        if 'signed_log1p' in actions and feat in train.columns:
            for part in (train, valid, test):
                part[feat] = np.sign(part[feat]) * np.log1p(np.abs(part[feat]))

    return train, valid, test
