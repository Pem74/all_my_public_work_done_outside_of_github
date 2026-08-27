import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV, ShuffleSplit, ParameterGrid
from sklearn.metrics import make_scorer, silhouette_score
import pickle


# conformation comparaison : idée : mesurer l'amplitude entre le max et le min de toute distance entre paire d'atome de
# deux observations différentes

files_directory = './'


def distance_conformation(conf1, conf2):
    """conf1 & 2 have to be numpy tab[[x1, y1, z1], [x2, y2, z2], ...] where all sub tab are one atom from one observation
    (like t=0). Conf1 & 2 have to be from the same peptide.
    Calcul the difference between max and min distance between same atoms from the both conformations
    """
    d = np.sqrt(np.sum((conf1 - conf2)**2, axis = 1))
    return np.max(d) - np.min(d)

def distance_conformation2(conf1, conf2):
    """conf1 & 2 have to be numpy tab[[x1, y1, z1], [x2, y2, z2], ...] where all sub tab are one atom from one observation
    (like t=0). Conf1 & 2 have to be from the same peptide.
    Calcul the euclidian distance between atom1 and all other atoms for conf1 and conf2, then he calcul the difference
    between these 2 list of distances to see if some atoms have moved.
    But we need to do that with a second atom which is not in mouv with the first atom because in this case, if one atom
    mouv while keeping the same distance with the first atom, he cant keep the same distance with the second atom.
    So we do the same thing with a false atom (which is just a translation of the first atom to be sure that they are not
    mouving between them)
    and finally, we take the absolute value and then the maximum
    """
    return np.max(np.abs((np.linalg.norm(conf1 - conf1[0], axis=1) - np.linalg.norm(conf2 - conf2[0], axis=1)) -
                         (np.linalg.norm(conf1 - (conf1[0] + 1), axis=1) - np.linalg.norm(conf2 - (conf2[0] + 1), axis=1))))


def matrice_distance(df, file_name):
    """Céer la matrice (symétrique) de comparaison entre les conformations et la sauvegarde dans un .npy (attention 12Go par matrice)"""
    num_times = 4001  # number of states of the peptide (from 0 ps to 200000 ps)
    time_interval = 5  # (ps), each state of the peptide
    num_atoms = len(df[df["Time"] == 0])  # number of atoms which form the peptide (between 79 and 89)

    # Créer un tableau 3D pour stocker les coordonnées de toutes les conformations
    conformations = np.zeros((num_times, num_atoms, 3))  # 3 for x, y and z
    for t in range(num_times):
        conformations[t] = df[df["Time"] == t * time_interval][["x", "y", "z"]].values
    del df

    M_distance = np.zeros((num_times, num_times))
    for t1 in range(num_times - 1):
        print(t1)
        conf1_1 = conformations[t1] - conformations[0]
        for t2 in range(t1 + 1, num_times):
            #M_distance[t1, t2] = distance_conformation(conformations[t1], conformations[t2])
            #d = np.sqrt(np.sum((conformations[t1] - conformations[t2]) ** 2, axis=1))  # to optimise calculs, directly here without function
            #M_distance[t1, t2] = np.max(d) - np.min(d)

            conf2_2 = conformations[t2] - conformations[0]
            M_distance[t1, t2] = np.max(np.abs((np.linalg.norm(conf1_1, axis=1) - np.linalg.norm(conf2_2, axis=1)) -
                                               (np.linalg.norm(conf1_1 - 1, axis=1) - np.linalg.norm(conf2_2 - 1, axis=1))))
            M_distance[t2, t1] = M_distance[t1, t2]

    matrice_name = file_name.split(".")[0] + "_matrice_distance.npy"
    np.save(matrice_name, M_distance)  # save the matrice
    # utiliser np.load(matrice_name) to load the tab

    image_name = matrice_name.split(".")[0] + "_heatmap.png"
    plt.figure()
    plt.imshow(M_distance, cmap='jet', interpolation='none')
    plt.colorbar()
    plt.title("Heatmap de la matrice des distances")
    plt.xlabel("Temps (ps)")
    plt.ylabel("Temps (ps)")
    plt.savefig(image_name)  # save the heatmap
    plt.close()

    return M_distance


def DBSCAN_conformation(file_name, best_dbscan, M_distance):
    """Useful to just use directly the best model of dbscan instead of search the best model among a lot of others"""
    labels = best_dbscan.fit_predict(M_distance)  # calcul the labels of each state

    labels_name = file_name.split(".")[0] + "_DBSCAN_labels.npy"
    np.save(labels_name, labels)  # save the labels
    # utiliser np.load(labels_name) to load the tab
    return labels


def best_DBSCAN_conformation(file_name, M_distance):
    """research of the best model of dbscan among the determined possibilities of hyperparameters"""

    # Fonction pour évaluer DBSCAN avec le score de silhouette
    def evaluate_dbscan(params, X):
        dbscan = DBSCAN(**params, metric="precomputed")
        labels = dbscan.fit_predict(X)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            # Gérer le cas où DBSCAN ne trouve pas assez de clusters
            return -1  # ou autre traitement pour les paramètres invalides
        silhouette = silhouette_score(X, labels)
        return silhouette

    param_grid = {  # each possibilities for the dbscan models to test
        'eps': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # epsilon, the max distance to be neighbors
        'min_samples': [5, 25, 125]  # min number of neighbors
    }
    dbscan = DBSCAN(metric="precomputed")  # we use directly the distances matrix instead of a function prewritten

    best_score = -1
    best_params = {}

    # Itérer sur tous les paramètres à tester
    for params in ParameterGrid(param_grid):
        score = evaluate_dbscan(params, M_distance)
        print(f"Parameters: {params} - Silhouette Score: {score}")
        if score > best_score:
            best_score = score
            best_params = params

    best_dbscan = DBSCAN(**best_params, metric="precomputed")
    labels = best_dbscan.fit_predict(M_distance)  # generation of labels

    labels_name = file_name.split(".")[0] + "_DBSCAN_labels.npy"
    np.save(labels_name, labels)
    # utiliser np.load(labels_name) to load the tab

    with open('dbscan_model.pkl', 'wb') as f:
        pickle.dump(best_dbscan, f)  # save of the best dbscan_model

    return labels, best_dbscan

def gen_labels(files_directory):
    """main function of the python file"""
    for file_name in os.listdir(files_directory):
        if file_name.endswith('reduced2.csv'):  # each csv
            print(file_name)
            df = pd.read_csv(file_name)

            labels_name = file_name.split(".")[0] + "_DBSCAN_labels.npy"
            if os.path.isfile(os.path.join(files_directory, labels_name)):  # search of labels to check if already gerenated
                labels = np.load(labels_name)  # load
            else:
                matrice_name = file_name.split(".")[0] + "_matrice_distance.npy"
                if os.path.isfile(os.path.join(files_directory, matrice_name)):  # check if distances matrix already generated
                    M_distance = np.load(matrice_name)  # load
                else:
                    M_distance = matrice_distance(df, file_name)  # generation

                if os.path.isfile(os.path.join(files_directory, 'dbscan_model2.pkl')):  # check if dbscan model already generated
                    with open('dbscan_model2.pkl', 'rb') as f:
                        best_dbscan = pickle.load(f)  # load
                    labels = DBSCAN_conformation(file_name, best_dbscan, M_distance)
                else:
                    labels, best_dbscan = best_DBSCAN_conformation(file_name, M_distance)  # generation

                del M_distance

            plt.figure()  # graph of labels
            image_name = labels_name.split(".")[0] + "_graph.png"
            plt.plot(np.arange(0, 20001, 5), labels, marker='o', linestyle='-', markersize=5)
            plt.title('Labels de cluster au fil du temps')
            plt.xlabel('Temps (ps)')
            plt.ylabel('Label de cluster')
            plt.savefig(image_name)  # save
            plt.close()

            del labels
            del df

gen_labels(files_directory)