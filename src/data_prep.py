"""
This python file's for preprocessing and data Scrubbing
"""

#===========import  moduls==============
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_dir = Path(__file__).resolve().parent.parent
#=============functions=================
def loader(show_print=False,return_test=False):
    #==============read CSV=============
    data_path = base_dir/"DataSet"/"creditcard.csv"
    credits_df = pd.read_csv(data_path)

    #=======analyzing dataset========
    if show_print:
        print(credits_df.head(3))
        print("_" * 20)
        print(credits_df.info())
        print("_" * 20)
        print(credits_df.describe())

        #==========number of Nan and Duplicates=========
        print("_" * 20)
        print(f"Sum of nan \n{credits_df.isna().sum()}")
        print("_" * 20)
        print(f"Duplicates samples {credits_df.duplicated().sum()}")
        print("_" * 20)
        print(credits_df["Class"].value_counts(normalize=True))
    credits_df.drop_duplicates(inplace=True)

    #==========split==============
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        credits_df.iloc[:, :-1].to_numpy(),
        credits_df.iloc[:, -1].to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=credits_df.iloc[:, -1],
    )
    #============Extract Validation==============
    Xtrain, Xval, ytrain, yval = train_test_split(
        Xtrain, ytrain, test_size=0.2, random_state=42, stratify=ytrain
    )

    #==========scaler==============
    scaler = StandardScaler()
    scaler.fit(Xtrain)

    if show_print:
        print("_" * 20)
        print(f"Xtrain {Xtrain.shape},\
              Xval {Xval.shape},\
             Xtest {Xtest.shape}, \
             ytrain {ytrain.shape}, \
              yval {yval.shape},\
             ytest {ytest.shape}")

    #===============Save-Scaler==============
    model_path = base_dir/"models"
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_path/"scaler.pkl")

    #==========Optional Return===============
    if return_test:
        return Xtest,ytest
    else:
        return (
            Xtrain,
            Xval,
            ytrain,
            yval,

        )


#===============Test Unit=================
if __name__ == "__main__":
    Xtrain, Xval, ytrain, yval = loader(show_print=False)
    print("Data processed successfully!")
