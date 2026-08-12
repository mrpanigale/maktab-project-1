"""
This python file's for preproccesing and data Scrubbing
"""

# import  moduls
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def loader(show_print=False):
    # load dataset from DataSet folder in csv format
    credits_df = pd.read_csv(r"E:\MLprojects\maktap-project-1\DataSet\creditcard.csv")

    # analyzing dataset
    if show_print:
        print(credits_df.head(3))
        print("_" * 20)
        print(credits_df.info())
        print("_" * 20)
        print(credits_df.describe())

        # check for nan
        print("_" * 20)
        print(f"Sum of nan \n{credits_df.isna().sum()}")
        print("_" * 20)
        print(f"Duplicates samples {credits_df.duplicated().sum()}")
        print("_" * 20)
        print(credits_df["Class"].value_counts(normalize=True))
    credits_df.drop_duplicates(inplace=True)

    # split data to train and test
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        credits_df.iloc[:, :-1].to_numpy(),
        credits_df.iloc[:, -1].to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=credits_df.iloc[:, -1],
    )

    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain_scaled = scaler.transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)
    if show_print:
        print("_" * 20)
        print(f"Xtrain {Xtrain_scaled.shape},\
             Xtest {Xtest_scaled.shape}, \
             ytrain {ytrain.shape}, \
             ytest {ytest.shape}")
    # save scaler for next uses
    if not os.path.exists(r"E:\MLprojects\maktap-project-1\models"):
        os.makedirs(r"E:\MLprojects\maktap-project-1\models")
    # save scaler object in models folder
    joblib.dump(scaler, r"E:\MLprojects\maktap-project-1\models\scaler.pkl")
    return Xtrain_scaled, Xtest_scaled, ytrain, ytest


if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest = loader(show_print=False)
    print("Data processed successfully!")
