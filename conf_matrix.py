import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    y_pred_reg = np.load('y_pred_reg.p')
    y_pred_automl = np.load('y_pred_automl.p')
    y_test = np.load('y_test.p')

    f, axes = plt.subplots(1, 2)
    ax = axes[0]
    cmat = confusion_matrix(y_test, y_pred_reg)
    sns.heatmap(cmat, annot=True, linewidth=3, fmt='d', cmap='Reds', ax=ax, xticklabels=np.unique(y_test).astype(int), yticklabels=np.unique(y_test).astype(int), linecolor='k', mask=cmat==0, vmax=340, vmin=0)
    ax.set_xlabel('Predicted quality')
    ax.set_ylabel('Expected quality')
    ax.set_title('Linear regression')

    ax = axes[1]
    cmat = confusion_matrix(y_test, y_pred_automl)
    sns.heatmap(cmat, annot=True, linewidth=3, fmt='d', cmap='Reds', ax=ax, xticklabels=np.unique(y_test).astype(int), yticklabels=np.unique(y_test).astype(int), linecolor='k', mask=cmat==0, vmax=340, vmin=0)
    ax.set_xlabel('Predicted quality')
    ax.set_ylabel('Expected quality')
    ax.set_title('auto-sklearn')

    plt.show()
