# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection

from config import INPUT_FILE
from config import COLUMN_NAMES

if __name__ == "__main__":

    # Read the training data
    df = pd.read_csv(INPUT_FILE, names=COLUMN_NAMES)

    # create a new column kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.income_bracket.values

    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    # save the new csv with kfold column
    df.to_csv('../input/adult-training_folds.csv', index=False)
