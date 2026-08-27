import pandas as pd
import numpy as np
import os


files_directory = './'

Peptide = []
Nb_conformations = []
Nb_outliers = []
Average_nb_state_per_conf = []
Actif = []

activity_map = {
            'APGVGV': 0,  # inactif
            'EGFEPG': 1,  # actif
            'GVAPGV': 1,
            'GVGVAP': 0,
            'LGTIPG': 1,
            'PGAIPG': 1,
            'PGAYPG': 1,
            'PGVGVA': 0,
            'VAPGVG': 0,
            'VGLAPG': 1,
            'VGVAPG': 1,
            'VVGPGA': 0
        }

for file_name in os.listdir(files_directory):
    if file_name.endswith('reduced2.csv'):
        print(file_name)

        """
        df = pd.read_csv(file_name)
        
        print(df.info())
        print(df.describe())
        print(df.head())

        print(df.isna().sum())  # pas de valeur manquante
        print(df.duplicated().sum())  # pas de doublons

        #  vérifions s'il y a des outliers et leur nombre
        print("x min | max : ", df["x"].min(), " : ", (df['x'] < 0).sum(), " | ", df["x"].max(), " : ", (df['x'] > 4).sum())
        print("y min | max : ", df["y"].min(), " : ", (df['y'] < 0).sum(), " | ", df["y"].max(), " : ", (df['y'] > 4).sum())
        print("z min | max : ", df["z"].min(), " : ", (df['z'] < 0).sum(), " | ", df["z"].max(), " : ", (df['z'] > 4).sum())
        print("nombre outliers : ", ((df['x'] < 0) | (df['y'] < 0) | (df['z'] < 0) | (df['x'] > 4) | (df['y'] > 4) | (df['z'] > 4)).sum())
        # problème avec VGLAPG (de -2 à 6) et VVGPGA (de -0.2 à 4.2), positions < 0 et > 4
        # choix de pas les supprimer pour le moment car pas non plus abbérentes,
        # (nombre non négligeable de valeurs concernées, resp. 1000000 et 150000)

        #  labélisation ? pour le moment non
        #  features à supprimer ? pour le moment non
        #  scaling ? pour le moment non (faire attention à scaler par rapport à toutes les peptides, et pas juste une)

        print()
        print("-------------------------------------------------------------------------------------------------------")
        print()

        del df
        """



        labels_name = file_name.split(".")[0] + "_DBSCAN_labels.npy"
        if os.path.isfile(os.path.join(files_directory, labels_name)):  # search of labels to check if already gerenated
            labels = np.load(labels_name)  # load
            Peptide.append(file_name.split(".")[0])
            Nb_conformations.append(np.max(labels) + 1)
            Nb_outliers.append(np.count_nonzero(labels == -1))
            occurrences = np.bincount(labels[labels != -1])
            Average_nb_state_per_conf.append(np.mean(occurrences))
            if Peptide[-1].split("_")[0] in activity_map:
                Actif.append(activity_map[Peptide[-1].split("_")[0]])
            else:
                if "inactive" in Peptide[-1]:
                    Actif.append(0)  # car ce sera nécessairement une peptide générée
                else:
                    Actif.append(1)
        else:
            continue


print(Peptide)
peptides_list = []
df = pd.DataFrame({
    'Peptide': Peptide,
    'Nb_conformations': Nb_conformations,
    'Nb_outliers': Nb_outliers,
    'Average_nb_state_per_conf': Average_nb_state_per_conf,
    'Actif': Actif
})

df["Peptide"] = df["Peptide"].astype('string')
df["Nb_conformations"] = df["Nb_conformations"].astype(int)
df["Nb_outliers"] = df["Nb_outliers"].astype(int)
df["Average_nb_state_per_conf"] = df["Average_nb_state_per_conf"].astype(float)
df["Actif"] = df["Actif"].astype(int)

df.to_csv("peptides_clustered2.csv", index=False)