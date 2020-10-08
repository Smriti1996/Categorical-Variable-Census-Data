import pandas as pd
import joblib
import os
import argparse

from sklearn import metrics
from sklearn import preprocessing

import config
import model_dispatcher

def run(fold, model):

    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    
    # drop numerical columns
    df = df.drop(config.NUM_COLS, axis=1)
    
    # map targets to 0s and 1s
    target_mapping = {
        " <=50K": 0,
        " >50K": 1
    }
    
    df.loc[:, "income_bracket"] = df["income_bracket"].map(target_mapping)

    # all columns are features except income_bracket and kfolds columns
    features = [
        f for f in df.columns if f not in ('kfold', 'income_bracket')
        ]
    
    # fill all NaN values with NONE 
    # (as all columns are being converted to string
    # so it does not matter as all are categories)
    for col in features:
        # do not encode the numerical columns
        if col not in config.NUM_COLS:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # now it's time to label encode the features 
    for col in features:
        if col not in config.NUM_COLS:
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()

            # fit label encoder on all data
            lbl.fit(df[col])

            # transform all the data
            df.loc[:, col] = lbl.transform(df[col])
    
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get the training data
    x_train = df_train[features].values
    
    # get validation data
    x_valid = df_valid[features].values
    
    # initialize Logistic Regression model
    model = model_dispatcher.models[model]

    # fit model on training data (ohe)
    model.fit(x_train, df_train.income_bracket.values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income_bracket.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

    # save the model
    joblib.dump(
        model,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser=argparse.ArgumentParser()

    # add different arguments you need and their type
    parser.add_argument(
        "--fold",
        type=int
    )

    parser.add_argument(
        "--model",
        type=str
    )

    # read the arguments from command line
    args=parser.parse_args()

    # run the folds specified by command line arguments
    run(
        fold=args.fold,
        model=args.model
    )
