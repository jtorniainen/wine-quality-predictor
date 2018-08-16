import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


if __name__ == '__main__':
    automl_cv_r2 = 0.473575
    automl_nopp_cv_r2 = 0.473575

    automl_y = np.genfromtxt('automl_y.csv')
    automl_nopp_y = np.genfromtxt('automl_nopp_y.csv')
    y_test = np.genfromtxt('y_test.csv')

    # f, ax = plt.subplots(1)
    # sns.barplot([1, 2], [automl_cv_r2, automl_nopp_cv_r2], ax=ax)

    f, axes = plt.subplots(1, 2)
    ax = axes[0]
    sns.regplot(automl_y, y_test, y_jitter=.25, x_jitter=.25, scatter_kws={'s': 60, 'edgecolors': 'k'}, ax=ax)
    ax.set_xlim(3, 9)
    ax.set_ylim(3, 9)
    ax.grid()
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('True vs predicted values')

    r2_automl = r2_score(y_test, automl_y)
    mse_automl = mean_squared_error(y_test, automl_y)

    bbox = {'facecolor': 'wheat'}
    ax.text(0.05, .95, 'R2  = {:0.2f}\nMSE = {:0.2f}'.format(r2_automl, mse_automl), transform=ax.transAxes, ha='left', va='top', bbox=bbox)

    ax = axes[1]

    x = np.unique(y_test)

    y = []
    for value in x:
        idx = np.where(y_test == value)
        y.append(np.mean(y_test[idx] == automl_y[idx]) * 100)

    x = x.astype(int)
    sns.barplot(x, y, ax=ax, color='C0')
    ax.grid(axis='y', color='w', linewidth=3)
    ax.set_ylabel('%')
    ax.set_xlabel('Quality')
    ax.set_title('Accuracy per quality-level')
    ax.set_ylim(0, 100)
    total_acc = np.mean(y_test == automl_y) * 100
    for pos, percent in enumerate(y):
        ax.text(pos, percent, '{:d}%'.format(int(percent)), ha='center', va='bottom', color='C0')
    ax.text(2.5, 100, '\nTotal accuracy = {:d}%'.format(int(total_acc)), ha='center', va='top', color='C0')
    f.suptitle('auto-sklearn test set performance', fontweight='bold')

    plt.show()
