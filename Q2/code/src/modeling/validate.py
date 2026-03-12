from __future__ import annotations


def _metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    y_pred = (y_prob >= threshold).astype(int)
    return {
        'AUC': float(roc_auc_score(y_true, y_prob)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_and_compare(train_df, valid_df, test_df, target_col: str, feature_sets: dict[str, list[str]], random_seed: int = 42):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    rows = []
    model_map = {
        'logistic': LogisticRegression(max_iter=200, random_state=random_seed),
        'hgbt': HistGradientBoostingClassifier(random_state=random_seed),
    }

    for fs_name, cols in feature_sets.items():
        X_train, y_train = train_df[cols], train_df[target_col]
        X_test, y_test = test_df[cols], test_df[target_col]

        imputer = SimpleImputer(strategy='median')
        X_train_i = imputer.fit_transform(X_train)
        X_test_i = imputer.transform(X_test)

        for model_name, model in model_map.items():
            model.fit(X_train_i, y_train)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_i)[:, 1]
            else:
                y_prob = model.decision_function(X_test_i)
            m = _metrics(y_test, y_prob)
            rows.append({'feature_set': fs_name, 'model': model_name, **m})

    return pd.DataFrame(rows)
