import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def read_log(filename):
    with open(filename, 'r') as file_h:
        content = file_h.readlines()

    preprocessors = []
    regressors = []
    rescalers = []
    idx = 0

    for line in content:
        if 'preprocessor:__choice__' in line:
            preprocessors.append(line.split('Value: ')[1].replace("'", '').strip())
        elif 'regressor:__choice__' in line:
            regressors.append(line.split('Value: ')[1].replace("'", '').strip())
        elif 'rescaling:__choice__' in line:
            rescalers.append(line.split('Value: ')[1].replace("'", '').strip())

    # just a quick check
    df = pd.DataFrame({'preprocessor': preprocessors, 'regressor': regressors, 'rescaling': rescalers})
    return df


if __name__ == '__main__':
    df = read_log('automl.log')
    # df = read_log('automl_no_pp.log')
    # df = pd.read_pickle('automl.p')
    # df = df.drop_duplicates()

    pre_x = 0
    reg_x = 20
    res_x = 10

    f, ax = plt.subplots(1)


    bbox = {'boxstyle': 'round', 'facecolor': 'wheat'}

    for idx, row in df.iterrows():
        y_pre = np.where(df.preprocessor.unique() == row.preprocessor)[0]
        y_res = np.where(df.rescaling.unique() == row.rescaling)[0]
        y_reg = np.where(df.regressor.unique() == row.regressor)[0]

        ax.plot([pre_x, res_x, reg_x], [y_pre, y_res, y_reg], 'k-', alpha=.3)

    for idx, preprocessor in enumerate(df.preprocessor.unique()):
        ax.plot(pre_x, idx, 'ko', markerfacecolor='w', markeredgecolor='k', markeredgewidth=1)
        ax.text(pre_x, idx, preprocessor, ha='right', va='center', bbox=bbox)

    for idx, rescaler in enumerate(df.rescaling.unique()):
        ax.plot(res_x, idx, 'ko', markerfacecolor='w', markeredgecolor='k', markeredgewidth=1)
        ax.text(res_x, idx, rescaler, ha='center', va='center', bbox=bbox)

    for idx, regressor in enumerate(df.regressor.unique()):
        ax.plot(reg_x, idx, 'ko', markerfacecolor='w', markeredgecolor='k', markeredgewidth=1)
        ax.text(reg_x, idx, regressor, ha='left', va='center', bbox=bbox)

    ax.text(reg_x, len(df.regressor.unique()), 'Regressor', ha='left', va='center', fontweight='bold')
    ax.text(res_x, len(df.rescaling.unique()), 'Rescaling', ha='center', va='center', fontweight='bold')
    ax.text(pre_x, len(df.preprocessor.unique()), 'Preprocessor', ha='right', va='center', fontweight='bold')


    ax.set_xlim(-20, 30)
    ax.axis('off')
    plt.show()

