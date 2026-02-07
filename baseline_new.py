"""
XGBoost Baseline for IEEE-CIS Fraud Detection
Based on 1st place solution by Chris Deotte (LB 0.96)
"""

import numpy as np
import pandas as pd
import os
import gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb


STR_TYPE = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]
STR_TYPE += [
    "id-12",
    "id-15",
    "id-16",
    "id-23",
    "id-27",
    "id-28",
    "id-29",
    "id-30",
    "id-31",
    "id-33",
    "id-34",
    "id-35",
    "id-36",
    "id-37",
    "id-38",
]

BASE_COLS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

V_COLS = [
    1,
    3,
    4,
    6,
    8,
    11,
    13,
    14,
    17,
    20,
    23,
    26,
    27,
    30,
    36,
    37,
    40,
    41,
    44,
    47,
    48,
    54,
    56,
    59,
    62,
    65,
    67,
    68,
    70,
    76,
    78,
    80,
    82,
    86,
    88,
    89,
    91,
    107,
    108,
    111,
    115,
    117,
    120,
    121,
    123,
    124,
    127,
    129,
    130,
    136,
    138,
    139,
    142,
    147,
    156,
    160,
    162,
    165,
    166,
    169,
    171,
    173,
    175,
    176,
    178,
    180,
    182,
    185,
    187,
    188,
    198,
    203,
    205,
    207,
    209,
    210,
    215,
    218,
    220,
    221,
    223,
    224,
    226,
    228,
    229,
    234,
    235,
    238,
    240,
    250,
    252,
    253,
    257,
    258,
    260,
    261,
    264,
    266,
    267,
    271,
    274,
    277,
    281,
    283,
    284,
    285,
    286,
    289,
    291,
    294,
    296,
    297,
    301,
    303,
    305,
    307,
    309,
    310,
    314,
    320,
]

FEATURES_TO_REMOVE_TIME_CONSISTENCY = [
    "C3",
    "M5",
    "id_08",
    "id_33",
    "card4",
    "id_07",
    "id_14",
    "id_21",
    "id_30",
    "id_32",
    "id_34",
]
FEATURES_TO_REMOVE_TIME_CONSISTENCY += ["id_" + str(x) for x in range(22, 28)]
D_COLS_TO_REMOVE = ["D6", "D7", "D8", "D9", "D12", "D13", "D14"]


def get_dtypes():
    """Get column data types for loading."""
    cols = BASE_COLS + ["V" + str(x) for x in V_COLS]
    dtypes = {}
    for c in (
        cols
        + ["id_0" + str(x) for x in range(1, 10)]
        + ["id_" + str(x) for x in range(10, 34)]
        + ["id-0" + str(x) for x in range(1, 10)]
        + ["id-" + str(x) for x in range(10, 34)]
    ):
        dtypes[c] = "float32"
    for c in STR_TYPE:
        dtypes[c] = "category"
    return dtypes


def load_data(input_dir):
    """Load train and test data."""
    dtypes = get_dtypes()
    cols = BASE_COLS + ["V" + str(x) for x in V_COLS]

    X_train = pd.read_csv(
        os.path.join(input_dir, "train_transaction.csv"),
        index_col="TransactionID",
        dtype=dtypes,
        usecols=cols + ["isFraud"],
    )
    train_id = pd.read_csv(
        os.path.join(input_dir, "train_identity.csv"),
        index_col="TransactionID",
        dtype=dtypes,
    )
    X_train = X_train.merge(train_id, how="left", left_index=True, right_index=True)

    X_test = pd.read_csv(
        os.path.join(input_dir, "test_transaction.csv"),
        index_col="TransactionID",
        dtype=dtypes,
        usecols=cols,
    )
    test_id = pd.read_csv(
        os.path.join(input_dir, "test_identity.csv"),
        index_col="TransactionID",
        dtype=dtypes,
    )
    fix = {o: n for o, n in zip(test_id.columns, train_id.columns)}
    test_id.rename(columns=fix, inplace=True)
    X_test = X_test.merge(test_id, how="left", left_index=True, right_index=True)

    y_train = X_train["isFraud"].copy()
    del train_id, test_id, X_train["isFraud"]
    gc.collect()

    print("Train shape", X_train.shape, "test shape", X_test.shape)
    return X_train, X_test, y_train


def normalize_d_columns(X_train, X_test):
    """Normalize D columns to stop them from increasing with time."""
    for i in range(1, 16):
        if i in [1, 2, 3, 5, 9]:
            continue
        X_train["D" + str(i)] = X_train[
            "D" + str(i)
        ] - X_train.TransactionDT / np.float32(24 * 60 * 60)
        X_test["D" + str(i)] = X_test["D" + str(i)] - X_test.TransactionDT / np.float32(
            24 * 60 * 60
        )


def label_encode_and_reduce_memory(X_train, X_test):
    """Factorize categorical variables and shift numerics to be positive."""
    for i, f in enumerate(X_train.columns):
        if (str(X_train[f].dtype) == "category") | (X_train[f].dtype == "object"):
            df_comb = pd.concat([X_train[f], X_test[f]], axis=0)
            df_comb, _ = df_comb.factorize(sort=True)
            if df_comb.max() > 32000:
                print(f, "needs int32")
            X_train[f] = df_comb[: len(X_train)].astype("int16")
            X_test[f] = df_comb[len(X_train) :].astype("int16")
        elif f not in ["TransactionAmt", "TransactionDT"]:
            mn = np.min((X_train[f].min(), X_test[f].min()))
            X_train[f] -= np.float32(mn)
            X_test[f] -= np.float32(mn)
            X_train[f].fillna(-1, inplace=True)
            X_test[f].fillna(-1, inplace=True)


def encode_FE(df1, df2, cols):
    """Frequency encode columns."""
    for col in cols:
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col + "_FE"
        df1[nm] = df1[col].map(vc).astype("float32")
        df2[nm] = df2[col].map(vc).astype("float32")
        print(nm, ", ", end="")


def encode_LE(col, train, test, verbose=True):
    """Label encode a column."""
    df_comb = pd.concat([train[col], test[col]], axis=0)
    df_comb, _ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max() > 32000:
        train[nm] = df_comb[: len(train)].astype("int32")
        test[nm] = df_comb[len(train) :].astype("int32")
    else:
        train[nm] = df_comb[: len(train)].astype("int16")
        test[nm] = df_comb[len(train) :].astype("int16")
    del df_comb
    gc.collect()
    if verbose:
        print(nm, ", ", end="")


def encode_AG(
    main_columns, uids, aggregations, train_df, test_df, fillna=True, usena=False
):
    """Group aggregation with mean/std."""
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column + "_" + col + "_" + agg_type
                temp_df = pd.concat(
                    [train_df[[col, main_column]], test_df[[col, main_column]]]
                )
                if usena:
                    temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan
                temp_df = (
                    temp_df.groupby([col])[main_column]
                    .agg([agg_type])
                    .reset_index()
                    .rename(columns={agg_type: new_col_name})
                )
                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_df[new_col_name] = train_df[col].map(temp_df).astype("float32")
                test_df[new_col_name] = test_df[col].map(temp_df).astype("float32")

                if fillna:
                    train_df[new_col_name].fillna(-1, inplace=True)
                    test_df[new_col_name].fillna(-1, inplace=True)

                print("'" + new_col_name + "'", ", ", end="")


def encode_CB(col1, col2, df1, df2):
    """Combine two columns."""
    nm = col1 + "_" + col2
    df1[nm] = df1[col1].astype(str) + "_" + df1[col2].astype(str)
    df2[nm] = df2[col1].astype(str) + "_" + df2[col2].astype(str)
    encode_LE(nm, df1, df2, verbose=False)
    print(nm, ", ", end="")


def encode_AG2(main_columns, uids, train_df, test_df):
    """Group aggregation with nunique count."""
    for main_column in main_columns:
        for col in uids:
            comb = pd.concat(
                [train_df[[col] + [main_column]], test_df[[col] + [main_column]]],
                axis=0,
            )
            mp = comb.groupby(col)[main_column].agg(["nunique"])["nunique"].to_dict()
            train_df[col + "_" + main_column + "_ct"] = (
                train_df[col].map(mp).astype("float32")
            )
            test_df[col + "_" + main_column + "_ct"] = (
                test_df[col].map(mp).astype("float32")
            )
            print(col + "_" + main_column + "_ct, ", end="")


def feature_engineering(X_train, X_test):
    """Create new features."""
    X_train["cents"] = (
        X_train["TransactionAmt"] - np.floor(X_train["TransactionAmt"])
    ).astype("float32")
    X_test["cents"] = (
        X_test["TransactionAmt"] - np.floor(X_test["TransactionAmt"])
    ).astype("float32")
    print("cents, ", end="")

    encode_FE(X_train, X_test, ["addr1", "card1", "card2", "card3", "P_emaildomain"])
    encode_CB("card1", "addr1", X_train, X_test)
    encode_CB("card1_addr1", "P_emaildomain", X_train, X_test)
    encode_FE(X_train, X_test, ["card1_addr1", "card1_addr1_P_emaildomain"])
    encode_AG(
        ["TransactionAmt", "D9", "D11"],
        ["card1", "card1_addr1", "card1_addr1_P_emaildomain"],
        ["mean", "std"],
        X_train,
        X_test,
        usena=True,
    )


def apply_feature_selection(X_train, X_test):
    """Remove features that fail time consistency test."""
    cols = list(X_train.columns)
    cols.remove("TransactionDT")

    for c in D_COLS_TO_REMOVE:
        cols.remove(c)

    for c in FEATURES_TO_REMOVE_TIME_CONSISTENCY:
        cols.remove(c)

    X_train = X_train[cols]
    X_test = X_test[cols]

    print("NOW USING THE FOLLOWING", len(cols), "FEATURES.")
    return X_train, X_test, cols


def create_month_feature(X_train, X_test):
    """Create month feature from TransactionDT."""
    import datetime

    START_DATE = datetime.datetime.strptime("2017-11-30", "%Y-%m-%d")
    X_train["DT_M"] = X_train["TransactionDT"].apply(
        lambda x: (START_DATE + datetime.timedelta(seconds=x))
    )
    X_train["DT_M"] = (X_train["DT_M"].dt.year - 2017) * 12 + X_train["DT_M"].dt.month

    X_test["DT_M"] = X_test["TransactionDT"].apply(
        lambda x: (START_DATE + datetime.timedelta(seconds=x))
    )
    X_test["DT_M"] = (X_test["DT_M"].dt.year - 2017) * 12 + X_test["DT_M"].dt.month


def train_xgb_model(X_train, y_train, cols, n_splits=6):
    """Train XGBoost with GroupKFold cross-validation."""
    oof = np.zeros(len(X_train))
    skf = GroupKFold(n_splits=n_splits)

    for i, (idxT, idxV) in enumerate(
        skf.split(X_train, y_train, groups=X_train["DT_M"])
    ):
        month = X_train.iloc[idxV]["DT_M"].iloc[0]
        print("Fold", i, "withholding month", month)
        print(" rows of train =", len(idxT), "rows of holdout =", len(idxV))

        clf = xgb.XGBClassifier(
            n_estimators=5000,
            max_depth=12,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            missing=-1,
            eval_metric="auc",
            tree_method="gpu_hist",
        )

        h = clf.fit(
            X_train[cols].iloc[idxT],
            y_train.iloc[idxT],
            eval_set=[(X_train[cols].iloc[idxV], y_train.iloc[idxV])],
            verbose=100,
            early_stopping_rounds=200,
        )

        oof[idxV] += clf.predict_proba(X_train[cols].iloc[idxV])[:, 1]

        del h, clf
        gc.collect()

    print("#" * 20)
    print("XGB OOF CV =", roc_auc_score(y_train, oof))
    return oof


def main():
    """Main training pipeline."""
    INPUT_DIR = "../input/ieee-fraud-detection"

    print("Loading data...")
    X_train, X_test, y_train = load_data(INPUT_DIR)

    print("\nNormalizing D columns...")
    normalize_d_columns(X_train, X_test)

    print("\nLabel encoding and reducing memory...")
    label_encode_and_reduce_memory(X_train, X_test)

    print("\nFeature engineering...")
    feature_engineering(X_train, X_test)

    print("\nApplying feature selection...")
    X_train, X_test, cols = apply_feature_selection(X_train, X_test)

    print("\nCreating month feature...")
    create_month_feature(X_train, X_test)

    print("\nTraining XGBoost model...")
    oof = train_xgb_model(X_train, y_train, cols)

    return oof


if __name__ == "__main__":
    main()
import argparse
