from __future__ import annotations


def add_redundancy_penalty(df_features, feature_cols: list[str], corr_threshold: float = 0.9):
    import numpy as np

    corr = df_features[feature_cols].corr(method='spearman').abs()
    penalty = {c: 0.0 for c in feature_cols}
    for i, c in enumerate(feature_cols):
        high_corr = int((corr.iloc[i] > corr_threshold).sum() - 1)
        penalty[c] = float(high_corr)
    return penalty


def compute_final_scores(diagnosis_df, redundancy_penalty: dict[str, float]):
    out = diagnosis_df.copy()
    out['redundancy_penalty'] = out['feature'].map(redundancy_penalty).fillna(0.0)
    out['data_quality_penalty'] = out['missing_rate'] + out['outlier_rate']

    alpha, beta, gamma, delta = 0.55, 0.30, 0.10, 0.05
    out['final_score'] = (
        alpha * out['predictive_score']
        + beta * out['stability_score']
        - gamma * out['redundancy_penalty']
        - delta * out['data_quality_penalty']
    )
    return out.sort_values('final_score', ascending=False).reset_index(drop=True)
