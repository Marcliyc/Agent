from __future__ import annotations


def run_feature_diagnosis(train_df, feature_cols: list[str], target_col: str):
    import numpy as np
    import pandas as pd

    rows = []
    y = train_df[target_col]
    for col in feature_cols:
        s = train_df[col]
        missing_rate = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        unique_ratio = float(nunique / max(len(s), 1))
        std = float(s.std(skipna=True) or 0.0)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outlier_rate = 0.0
        else:
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_rate = float(((s < low) | (s > high)).mean())
        skewness = float(s.skew(skipna=True) or 0.0)

        # temporal stability: mean std across trade_date groups
        if 'trade_date' in train_df.columns:
            grp = train_df.groupby('trade_date')[col].mean()
            stability = float(1.0 / (1.0 + (grp.std(skipna=True) or 0.0)))
        else:
            stability = 0.5

        corr = train_df[[col, target_col]].dropna().corr(method='spearman').iloc[0, 1]
        predictive = float(abs(corr) if not np.isnan(corr) else 0.0)

        diagnosis = 'healthy'
        if missing_rate > 0.4 or unique_ratio < 0.001:
            diagnosis = 'high_risk'
        elif outlier_rate > 0.15 or abs(skewness) > 2.0:
            diagnosis = 'needs_transform'

        rows.append(
            {
                'feature': col,
                'missing_rate': missing_rate,
                'outlier_rate': outlier_rate,
                'skewness': skewness,
                'std': std,
                'unique_ratio': unique_ratio,
                'predictive_score': predictive,
                'stability_score': stability,
                'diagnosis': diagnosis,
            }
        )
    return pd.DataFrame(rows).sort_values('predictive_score', ascending=False).reset_index(drop=True)
