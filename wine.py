# Jyväskylä 28th Summer School
# COM2 group work

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest


if __name__ == '__main__':

    # Load data from CSV
    df_red = pd.read_csv('winequality-red.csv', sep=';')
    df_white = pd.read_csv('winequality-white.csv', sep=';')

    # Concatenate red and white wines into a single DataFrame
    df_red['type'] = ['red'] * df_red.shape[0]
    df_white['type'] = ['white'] * df_white.shape[0]
    df = pd.concat((df_red, df_white), axis=0)

    # Visualize distribution of each feature
    # https://seaborn.pydata.org/generated/seaborn.distplot.html
    f, axes = plt.subplots(3, 4)
    for ax, var in zip(axes.flatten(), df.drop(['quality', 'type'], axis=1).columns):
        sns.distplot(df[df.type == 'red'][var], ax=ax)
        sns.distplot(df[df.type == 'white'][var], ax=ax)
        # sns.distplot(df[var], ax=ax)
        # Test if the feature is normally distributed
        print('{:<20} k2={:10.2f} p={:0.2f}'.format(var, *normaltest(df[df.type == 'red'][var].get_values())))
        print('{:<20} k2={:10.2f} p={:0.2f}'.format(var, *normaltest(df[df.type == 'white'][var].get_values())))

    # Visualize correlations between features
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    f, ax = plt.subplots(1)
    feature_corr = df[df.type == 'red'].drop(['quality', 'type'], axis=1).corr()
    sns.heatmap(feature_corr, annot=True, mask=feature_corr.abs() <= .3, cmap='RdBu_r', fmt='0.1f', linewidths=1, ax=ax, vmax=1, vmin=-1)
    ax.set_title('red')

    f, ax = plt.subplots(1)
    feature_corr = df[df.type == 'white'].drop(['quality', 'type'], axis=1).corr()
    sns.heatmap(feature_corr, annot=True, mask=feature_corr.abs() <= .3, cmap='RdBu_r', fmt='0.1f', linewidths=1, ax=ax, vmax=1, vmin=-1)
    ax.set_title('white')

    plt.show()
