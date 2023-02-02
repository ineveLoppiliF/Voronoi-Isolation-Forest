from datetime import datetime
from matplotlib import pyplot as plt
from numpy import full, genfromtxt, nan, ndarray, save
from os import makedirs
from PreferenceIForest import IForest
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, roc_curve
from time import time


# region Define parameters
datasets_folder: str = '../datasets/ifor/'
results_path: str = '../experiments/ifor/' + datetime.now().strftime('%Y_%m_%d_%H%M%S') + '/'
# datasets: list[str] = ['Annthyroid.csv', 'Arrhythmia.csv', 'Breastw.csv', 'ForestCover.csv', 'hbk.csv', 'Http.csv',
#                        'Ionosphere.csv', 'Mammography.csv', 'Mulcross.csv', 'Pendigits.csv', 'Pima.csv',
#                        'Satellite.csv', 'Shuttle.csv', 'Smtp.csv', 'wood.csv']
datasets: list[str] = ['Annthyroid.csv', 'Arrhythmia.csv', 'Breastw.csv', 'hbk.csv', 'Ionosphere.csv','Mammography.csv',
                       'Pendigits.csv', 'Pima.csv', 'Satellite.csv', 'Shuttle.csv', 'Smtp.csv', 'wood.csv']
branching_factors: list[int] = [2, 4, 8, 16, 32, 64]
metrics: list[str] = ['tanimoto', 'ruzicka', 'euclidean', 'cityblock', 'cosine', 'hamming', 'jaccard']
ivor_params: dict = {'iforest_type': 'voronoiiforest',
                     'num_trees': 100,
                     'max_samples': 256,
                     'n_jobs': 1}
ifor_params: dict = {'n_estimators': 100,
                     'max_samples': 256,
                     'n_jobs': 1}
n_iterations: int = 5
# endregion

# Create output folder
makedirs(results_path, exist_ok=True)

# Create arrays to store results of IFOR
results_ifor: ndarray = full(shape=(n_iterations, len(datasets)), fill_value=nan, dtype=float)
train_time_ifor: ndarray = full(shape=(n_iterations, len(datasets)), fill_value=nan, dtype=float)
test_time_ifor: ndarray = full(shape=(n_iterations, len(datasets)), fill_value=nan, dtype=float)

# Create arrays to store results of IVOR
results_ivor: ndarray = full(shape=(len(metrics), n_iterations, len(datasets), len(branching_factors)), fill_value=nan,
                             dtype=float)
train_time_ivor: ndarray = full(shape=(len(metrics), n_iterations, len(datasets), len(branching_factors)), fill_value=nan,
                                dtype=float)
test_time_ivor: ndarray = full(shape=(len(metrics), n_iterations, len(datasets), len(branching_factors)), fill_value=nan,
                               dtype=float)

for iteration in range(n_iterations):
    print('Iteration #' + str(iteration + 1))
    for dataset_index, _ in enumerate(datasets):
        print(str(datasets[dataset_index]))

        # region Create dataset
        dataset: ndarray = genfromtxt(datasets_folder + datasets[dataset_index], delimiter=',')

        # Remove heading and id column
        dataset: ndarray = dataset[1:, 1:]

        # Compose dataset
        data: ndarray = dataset[:, :-1]
        inlier_mask: ndarray = ~dataset[:, -1].astype(dtype=bool)

        # endregion

        execution_name: str = datasets[dataset_index].split('.')[0]
        # region Isolation Forest

        # Instantiate IFOR
        ifor: IsolationForest = IsolationForest(**ifor_params)

        # Fit IFOR
        start_train_time: float = time()
        ifor.fit(data)
        end_train_time: float = time()

        # Score samples
        start_test_time: float = time()
        scores: ndarray = -ifor.score_samples(data)
        end_test_time: float = time()

        # endregion

        # region Plot performance

        # Compute FPR, TRP and AUC
        fpr, tpr, _ = roc_curve(~inlier_mask, scores)
        roc_auc: float = auc(fpr, tpr)

        # # Plot ROC curve together with AUC
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.3f' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('FPR')
        # plt.ylabel('TPR')
        # plt.axis('equal')
        # plt.title('Isolation Forest')
        # plt.legend(loc="lower right")
        # plt.tight_layout()
        # plt.savefig(results_path + 'ifor_roc_' + execution_name + '.svg', format='svg')
        # plt.close()
        # #plt.show()

        # endregion

        # Store results
        results_ifor[iteration, dataset_index]: float = roc_auc

        # Store times
        train_time_ifor[iteration, dataset_index]: float = end_train_time - start_train_time
        test_time_ifor[iteration, dataset_index]: float = end_test_time - start_test_time
        # Save results
        with open(results_path + 'roc_aucs_ifor.npy', 'wb') as f:
            save(f, results_ifor)
        with open(results_path + 'train_time_ifor.npy', 'wb') as f:
            save(f, train_time_ifor)
        with open(results_path + 'test_time_ifor.npy', 'wb') as f:
            save(f, test_time_ifor)

        for metric_index, _ in enumerate(metrics):
            print(str(metrics[metric_index]))
            ivor_params['metric']: str = metrics[metric_index]
            for branching_index, _ in enumerate(branching_factors):
                print(str(branching_factors[branching_index]))
                ivor_params['branching_factor']: float = branching_factors[branching_index]
                execution_name: str = str(ivor_params['iforest_type']) + '_' + datasets[dataset_index].split('.')[0] +\
                                      '_' + str(ivor_params['branching_factor']) + '_' + str(ivor_params['metric'])

                # region Isolation Voronoi Forest

                # Instantiate IVOR
                ivor: IForest = IForest.create(**ivor_params)

                # Fit IVOR
                start_train_time: float = time()
                ivor.fit(data)
                end_train_time: float = time()

                # Score samples
                start_test_time: float = time()
                scores: ndarray = ivor.score_samples(data)
                end_test_time: float = time()

                # endregion

                # region Plot performance

                # Compute FPR, TRP and AUC
                fpr, tpr, _ = roc_curve(~inlier_mask, scores)
                roc_auc: float = auc(fpr, tpr)

                # # Plot ROC curve together with AUC
                # lw = 2
                # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.3f' % roc_auc)
                # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                # plt.xlim([0.0, 1.0])
                # plt.ylim([0.0, 1.05])
                # plt.xlabel('FPR')
                # plt.ylabel('TPR')
                # plt.axis('equal')
                # plt.title('Isolation Voronoi Forest')
                # plt.legend(loc="lower right")
                # plt.tight_layout()
                # plt.savefig(results_path + 'ivor_roc_' + execution_name + '.svg', format='svg')
                # plt.close()
                # #plt.show()

                # endregion

                # Store results
                results_ivor[metric_index, iteration, dataset_index, branching_index]: float = roc_auc

                # Store times
                train_time_ivor[metric_index, iteration, dataset_index, branching_index]: float = end_train_time - start_train_time
                test_time_ivor[metric_index, iteration, dataset_index, branching_index]: float = end_test_time - start_test_time
                # Save results
                with open(results_path + 'roc_aucs_ivor.npy', 'wb') as f:
                    save(f, results_ivor)
                with open(results_path + 'train_time_ivor.npy', 'wb') as f:
                    save(f, train_time_ivor)
                with open(results_path + 'test_time_ivor.npy', 'wb') as f:
                    save(f, test_time_ivor)
# endregion
