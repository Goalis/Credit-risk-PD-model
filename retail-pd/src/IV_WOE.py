
import pandas as pd
import numpy as np
def iv_summary(df, target, features, bins=10, categorical_features=None):
    """
    Compute IV for a list of features and return a ranked summary table.

    df: DataFrame
    target: binary target column name
    features: list of continuous feature names to bin and evaluate
    bins: number of bins for continuous variables
    categorical_features: list of categorical feature names (evaluated without binning)
    """
    categorical_features = categorical_features or []
    results = []

    for feat in features:
        is_cat = feat in categorical_features
        try:
            _, iv = calculate_woe_iv(df, feat, target, bins=bins, is_categorical=is_cat)
            results.append({'feature': feat, 'iv': iv})
        except Exception as e:
            results.append({'feature': feat, 'iv': None, 'error': str(e)})

    summary = pd.DataFrame(results).sort_values('iv', ascending=False).reset_index(drop=True)
    summary['strength'] = summary['iv'].apply(_iv_strength)
    return summary

def _iv_strength(iv):
    if iv is None:
        return 'error'
    elif iv < 0.02:
        return 'not predictive'
    elif iv < 0.1:
        return 'weak'
    elif iv < 0.3:
        return 'medium'
    elif iv < 0.5:
        return 'strong'
    else:
        return 'suspicious (check leakage)'

def calculate_woe_iv(df, feature, target, bins=10, is_categorical=False):
    """
    Compute WOE and IV for a single feature.
    
    df: DataFrame containing feature and target
    feature: column name of the predictor
    target: column name of the binary target (1 = bad/default, 0 = good)
    bins: number of bins for continuous variables (ignored if categorical)
    is_categorical: True if feature should be treated as categorical (no binning)
    """
    data = df[[feature, target]].copy()

    if is_categorical:
        data['bin'] = data[feature].astype(str)
    else:
        # Quantile-based binning; duplicates='drop' handles skewed distributions
        data['bin'] = pd.qcut(data[feature], q=bins, duplicates='drop')

    grouped = data.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()

    # Add small constant to avoid division by zero / log(0)
    eps = 0.5
    grouped['dist_good'] = (grouped['good'] + eps) / (total_good + eps * len(grouped))
    grouped['dist_bad'] = (grouped['bad'] + eps) / (total_bad + eps * len(grouped))

    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
    grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']

    iv_total = grouped['iv'].sum()

    return grouped.reset_index(), iv_total


def apply_woe(df, feature, woe_table, is_categorical=False):
    """
    Map raw feature values onto their WOE values using a fitted woe_table
    (the first output of calculate_woe_iv).
    """
    mapping = woe_table.set_index('bin')['woe']

    if is_categorical:
        return df[feature].astype(str).map(mapping)
    else:
        # Re-cut using the same bin edges as the fitted table
        bins = pd.IntervalIndex(woe_table['bin'])
        cats = pd.cut(df[feature], bins=bins)
        return cats.map(mapping)