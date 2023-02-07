from matplotlib import pyplot as plt
from numpy import load, ndarray

results_path: str = '../experiments/ifor/2023_02_03_164920/'
datasets: list[str] = ['Annthyroid.csv', 'Arrhythmia.csv', 'Breastw.csv', 'ForestCover.csv', 'hbk.csv', 'Http.csv',
                       'Ionosphere.csv', 'Mammography.csv', 'Mulcross.csv', 'Pendigits.csv', 'Pima.csv',
                       'Satellite.csv', 'Shuttle.csv', 'Smtp.csv', 'wood.csv']
branching_factors: list[int] = [2, 4, 8, 16, 32]
metrics: list[str] = ['cityblock', 'euclidean', 'seuclidean', 'mahalanobis', 'cosine', 'correlation', 'hamming',
                      'jaccard', 'tanimoto', 'ruzicka', 'braycurtis', 'canberra', 'chebyshev']

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


roc_aucs_ivor_mean: ndarray = roc_aucs_ivor.mean(axis=1)
roc_aucs_ifor_mean: ndarray = roc_aucs_ifor.mean(axis=0)
# Plot ROC AUCs for branching factor 2
plt.scatter(datasets, roc_aucs_ivor_mean[0, :, 0], marker='^', label='ivor_tanimoto')
plt.scatter(datasets, roc_aucs_ivor_mean[1, :, 0], marker='<', label='ivor_ruzicka')
plt.scatter(datasets, roc_aucs_ivor_mean[2, :, 0], marker='o', label='ivor_euclidean')
plt.scatter(datasets, roc_aucs_ivor_mean[3, :, 0], marker='+', label='ivor_cityblock')
plt.scatter(datasets, roc_aucs_ivor_mean[4, :, 0], marker='x', label='ivor_cosine')
plt.scatter(datasets, roc_aucs_ivor_mean[5, :, 0], marker='D', label='ivor_hamming')
plt.scatter(datasets, roc_aucs_ivor_mean[6, :, 0], marker='v', label='ivor_jaccard')
plt.scatter(datasets, roc_aucs_ifor_mean[:], marker='s', label='ifor')
plt.ylim([0, 1.0])
plt.xticks(rotation=45)
plt.xlabel('Dataset')
plt.ylabel('ROC AUC')
plt.title('Branching factor = 2')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(results_path + 'roc_aucs_branching_2.svg', format='svg')
plt.close()

# Plot best ROC AUCs
plt.scatter(datasets, roc_aucs_ivor_mean[0, :, :].max(axis=1), marker='^', label='ivor_tanimoto')
plt.scatter(datasets, roc_aucs_ivor_mean[1, :, :].max(axis=1), marker='<', label='ivor_ruzicka')
plt.scatter(datasets, roc_aucs_ivor_mean[2, :, :].max(axis=1), marker='o', label='ivor_euclidean')
plt.scatter(datasets, roc_aucs_ivor_mean[3, :, :].max(axis=1), marker='+', label='ivor_cityblock')
plt.scatter(datasets, roc_aucs_ivor_mean[4, :, :].max(axis=1), marker='x', label='ivor_cosine')
plt.scatter(datasets, roc_aucs_ivor_mean[5, :, :].max(axis=1), marker='D', label='ivor_hamming')
plt.scatter(datasets, roc_aucs_ivor_mean[6, :, :].max(axis=1), marker='v', label='ivor_jaccard')
plt.scatter(datasets, roc_aucs_ifor_mean[:], marker='s', label='ifor')
plt.ylim([0, 1.0])
plt.xticks(rotation=45)
plt.xlabel('Dataset')
plt.ylabel('ROC AUC')
plt.title('Branching factor = best')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(results_path + 'roc_aucs_branching_best.svg', format='svg')
plt.close()

# # Plot ROC AUCs
# fig, ax = plt.subplots()
# for j, forest_metric in enumerate(forest_metric_tuple):
#     if forest_metric[1]:
#         ax.errorbar(branching_factors, roc_aucs[j, :, i, :].mean(axis=0), yerr=roc_aucs[j, :, i, :].std(axis=0),
#                     lw=2, label=forest_metric[0] + '_' + forest_metric[1])
#     else:
#         ax.errorbar(branching_factors, roc_aucs[j, :, i, :].mean(axis=0), yerr=roc_aucs[j, :, i, :].std(axis=0),
#                     lw=2, label=forest_metric[0])
# plt.ylim([0.5, 1.0])
# plt.xlabel('Branching factor')
# plt.ylabel('ROC AUC')
# plt.xscale('log')
# plt.title(dataset)
# plt.legend()
# plt.tight_layout()
# plt.savefig(results_path + dataset.split('.')[0] + '_roc_aucs.svg', format='svg')
# plt.close()
#
# # Plot train time
# fig, ax = plt.subplots()
# for j, forest_metric in enumerate(forest_metric_tuple):
#     if forest_metric[1]:
#         ax.errorbar(branching_factors, train_time[j, :, i, :].mean(axis=0), yerr=train_time[j, :, i, :].std(axis=0),
#                     lw=2, label=forest_metric[0] + '_' + forest_metric[1])
#     else:
#         ax.errorbar(branching_factors, train_time[j, :, i, :].mean(axis=0), yerr=train_time[j, :, i, :].std(axis=0),
#                     lw=2, label=forest_metric[0])
# plt.xlabel('Branching factor')
# plt.ylabel('Train time')
# plt.xscale('log')
# plt.title(dataset)
# plt.legend()
# plt.tight_layout()
# plt.savefig(results_path + dataset.split('.')[0] + '_train_time.svg', format='svg')
# plt.close()
#
# # Plot test time
# fig, ax = plt.subplots()
# for j, forest_metric in enumerate(forest_metric_tuple):
#     if forest_metric[1]:
#         ax.errorbar(branching_factors, test_time[j, :, i, :].mean(axis=0), yerr=test_time[j, :, i, :].std(axis=0),
#                     lw=2, label=forest_metric[0] + '_' + forest_metric[1])
#     else:
#         ax.errorbar(branching_factors, test_time[j, :, i, :].mean(axis=0), yerr=test_time[j, :, i, :].std(axis=0),
#                     lw=2, label=forest_metric[0])
# plt.xlabel('Branching factor')
# plt.ylabel('Test time')
# plt.xscale('log')
# plt.title(dataset)
# plt.legend()
# plt.tight_layout()
# plt.savefig(results_path + dataset.split('.')[0] + '_test_time.svg', format='svg')
# plt.close()
