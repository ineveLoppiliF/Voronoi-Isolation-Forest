from matplotlib import pyplot as plt
from numpy import load, ndarray

results_path: str = '../experiments/ifor/2023_01_27_140738/'
# datasets: list[str] = ['Annthyroid.csv', 'Arrhythmia.csv', 'Breastw.csv', 'ForestCover.csv', 'hbk.csv', 'Http.csv',
#                        'Ionosphere.csv', 'Mammography.csv', 'Mulcross.csv', 'Pendigits.csv', 'Pima.csv',
#                        'Satellite.csv', 'Shuttle.csv', 'Smtp.csv', 'wood.csv']
datasets: list[str] = ['Arrhythmia.csv', 'Breastw.csv', 'hbk.csv', 'Ionosphere.csv', 'Pima.csv', 'wood.csv']
branching_factors: list[int] = [2, 4, 8, 16, 32]
metrics: list[str] = ['tanimoto', 'ruzicka', 'euclidean', 'cityblock', 'cosine', 'hamming', 'jaccard']

with open(results_path + 'roc_aucs_ifor.npy', 'rb') as f:
    roc_aucs_ifor: ndarray = load(f)
with open(results_path + 'train_time_ifor.npy', 'rb') as f:
    train_time_ifor: ndarray = load(f)
with open(results_path + 'test_time_ifor.npy', 'rb') as f:
    test_time_ifor: ndarray = load(f)

with open(results_path + 'roc_aucs_ivor.npy', 'rb') as f:
    roc_aucs_ivor: ndarray = load(f)
with open(results_path + 'train_time_ivor.npy', 'rb') as f:
    train_time_ivor: ndarray = load(f)
with open(results_path + 'test_time_ivor.npy', 'rb') as f:
    test_time_ivor: ndarray = load(f)


for i, dataset in enumerate(datasets):
    # Plot ROC AUCs
    fig, ax = plt.subplots()
    for j, metric in enumerate(metrics):
        ax.errorbar(branching_factors, roc_aucs_ivor[j, :, i, :].mean(axis=0), yerr=roc_aucs_ivor[j, :, i, :].std(axis=0),
                    lw=2, label=metric)
    ax.errorbar([2], [roc_aucs_ifor[:, i].mean(axis=0)], yerr=[roc_aucs_ifor[:, i].std(axis=0)], lw=2,
                label='ifor (Manhattan LSH)')
    plt.ylim([0.5, 1.0])
    plt.xlabel('Branching factor')
    plt.ylabel('ROC AUC')
    plt.xscale('log')
    plt.title(dataset)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path + dataset + '_roc_aucs.svg', format='svg')
    plt.close()

    # Plot train time
    fig, ax = plt.subplots()
    for j, metric in enumerate(metrics):
        ax.errorbar(branching_factors, train_time_ivor[j, :, i, :].mean(axis=0), yerr=train_time_ivor[j, :, i, :].std(axis=0),
                    lw=2, label=metric)
    ax.errorbar([2], [train_time_ifor[:, i].mean(axis=0)], yerr=[train_time_ifor[:, i].std(axis=0)], lw=2,
                label='ifor (Manhattan LSH)')
    plt.xlabel('Branching factor')
    plt.ylabel('Train time')
    plt.xscale('log')
    plt.title(dataset)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path + dataset + '_train_time.svg', format='svg')
    plt.close()

    # Plot test time
    fig, ax = plt.subplots()
    for j, metric in enumerate(metrics):
        ax.errorbar(branching_factors, test_time_ivor[j, :, i, :].mean(axis=0), yerr=test_time_ivor[j, :, i, :].std(axis=0),
                    lw=2, label=metric)
    ax.errorbar([2], [test_time_ifor[:, i].mean(axis=0)], yerr=[test_time_ifor[:, i].std(axis=0)], lw=2,
                label='ifor (Manhattan LSH)')
    plt.xlabel('Branching factor')
    plt.ylabel('Test time')
    plt.xscale('log')
    plt.title(dataset)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path + dataset + '_test_time.svg', format='svg')
    plt.close()
