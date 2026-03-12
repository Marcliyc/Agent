from __future__ import annotations


def select_top_features(scored_df, top_k: int = 50, corr_threshold: float = 0.90, data_df=None):
    selected = []
    reasons = []

    for _, row in scored_df.iterrows():
        feat = row['feature']
        if len(selected) >= top_k:
            break

        redundant = False
        if data_df is not None and selected:
            corr = data_df[selected + [feat]].corr(method='spearman').abs()[feat].drop(feat)
            if len(corr) and corr.max() >= corr_threshold:
                redundant = True

        if redundant:
            continue
        selected.append(feat)
        reasons.append(
            f"score={row['final_score']:.4f}; predictive={row['predictive_score']:.4f}; stability={row['stability_score']:.4f}"
        )

    import pandas as pd

    return pd.DataFrame({'feature': selected, 'reason': reasons, 'rank': range(1, len(selected) + 1)})
