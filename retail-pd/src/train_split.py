import pandas as pd
from sklearn.model_selection import train_test_split


def split_pd_data(
    df,
    target_col,
    date_col=None,
    oot_cutoff=None,
    test_size=0.25,
    val_size=None,
    random_state=42
):
    """
    Split a PD modeling dataset into train/test sets.

    Parameters
    ----------
    df : DataFrame
        Full dataset.
    target_col : str
        Name of the binary target column (1 = bad/default).
    date_col : str, optional
        Name of the date/vintage column. If provided along with oot_cutoff,
        an out-of-time split is performed first.
    oot_cutoff : str or datetime, optional
        Observations with date_col >= oot_cutoff go into the OOT set.
        Ignored if date_col is None.
    test_size : float
        Proportion of the in-time data used for the test set.
    val_size : float, optional
        If provided, proportion of the in-time data used for a separate
        validation set (carved out before the test split).
    random_state : int
        Seed for reproducibility.

    """
    result = {}
    working_df = df.copy()

    # Step 1: out-of-time split, if requested
    if date_col is not None and oot_cutoff is not None:
        oot_mask = working_df[date_col] >= pd.to_datetime(oot_cutoff)
        result['oot'] = working_df[oot_mask].reset_index(drop=True)
        working_df = working_df[~oot_mask].reset_index(drop=True)

    # Step 2: optional validation carve-out (stratified)
    if val_size is not None:
        working_df, val_df = train_test_split(
            working_df,
            test_size=val_size,
            stratify=working_df[target_col],
            random_state=random_state
        )
        result['val'] = val_df.reset_index(drop=True)

    # Step 3: train/test split (stratified)
    train_df, test_df = train_test_split(
        working_df,
        test_size=test_size,
        stratify=working_df[target_col],
        random_state=random_state
    )
    result['train'] = train_df.reset_index(drop=True)
    result['test'] = test_df.reset_index(drop=True)

    return result