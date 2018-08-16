# Jyväskylä 28th Summer School
# COM2 group work

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from sklearn.model_selection import train_test_split

sns.set_context('talk')


if __name__ == '__main__':

    # Load data from CSV
    df = pd.read_csv('winequality-white.csv', sep=';')
    df_train, df_test = train_test_split(df, test_size=.20, random_state=1)


    # Visualize distribution of each feature
    # https://seaborn.pydata.org/generated/seaborn.distplot.html
    f, axes = plt.subplots(3, 4)
    for ax, var in zip(axes.flatten(), df.columns):
        if var == 'quality':
            sns.distplot(df_train[var], ax=ax, label='train')
            sns.distplot(df_test[var], ax=ax, label='test')
            ax.legend()
        else:
            sns.distplot(df_train[var], ax=ax)
            sns.distplot(df_test[var], ax=ax)

        ax.set_xlabel('')
        ax.set_title(var, fontweight='bold')
        ax.grid()

        # sns.distplot(df[var], ax=ax)
        # Test if the feature is normally distributed
        # print('{:<20} k2={:10.2f} p={:0.2f}'.format(var, *normaltest(df[df.type == 'red'][var].get_values())))
        # print('{:<20} k2={:10.2f} p={:0.2f}'.format(var, *normaltest(df[df.type == 'white'][var].get_values())))

    # Visualize correlations between features
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # f, ax = plt.subplots(1)
    # feature_corr = df[df.type == 'red'].drop(['quality', 'type'], axis=1).corr()
    # sns.heatmap(feature_corr, annot=True, mask=feature_corr.abs() <= .3, cmap='RdBu_r', fmt='0.1f', linewidths=1, ax=ax, vmax=1, vmin=-1)
    # ax.set_title('red')

    # f, ax = plt.subplots(1)
    # feature_corr = df[df.type == 'white'].drop(['quality', 'type'], axis=1).corr()
    # sns.heatmap(feature_corr, annot=True, mask=feature_corr.abs() <= .3, cmap='RdBu_r', fmt='0.1f', linewidths=1, ax=ax, vmax=1, vmin=-1)
    # ax.set_title('white')

    plt.show()
