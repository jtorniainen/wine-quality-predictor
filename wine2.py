# Jyväskylä 28th Summer School
# COM2 group work

import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    # Load data from CSV
    df = pd.read_csv('winequality-white.csv', sep=';')

    # Train-Test split
    df_train, df_test = train_test_split(df, test_size=.20, random_state=1)

    # 1. Manual

    # 2. Automatic


    # 3. Comparison of results from 1 and 2

