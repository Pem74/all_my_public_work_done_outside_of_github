import pandas as pd
import os


# Définir une fonction pour extraire les données d'un fichier .gro
def extract_data(file_path, peptide_name):
    data = []
    time = -5  # temps en ps
    with open(file_path, 'r') as file:
        lines = file.readlines()  # tableau de toutes les lignes du fichier
        for line in lines:
            if not line.startswith("    "):  # seuls les lignes d'atomes commencent par 4 espaces
                if line.startswith("G"):  # 3 lignes par observation ne sont pas des atomes, n'ajoute 5 à time qu'une seule fois
                    time += 5
                continue
            else:
                atom = [peptide_name, time]  # pour un atome
                atom.extend(line.split())  # ajout des autres caractéristiques,
                data.append(atom)  # ajout de l'atome
    return data


files_directory = '../Trajectoires_Peptides/'
num = 0

for file_name in os.listdir(files_directory):
    num += 1
    if file_name.endswith('.gro'):
        file_path = os.path.join(files_directory, file_name)
        peptide_name = file_name.split('_')[0]
        peptide_data = extract_data(file_path, peptide_name)
        print("Ficher",  num, "(", file_name, ")", "extrait.")

        # Créer un DataFrame pandas à partir des données
        df = pd.DataFrame(peptide_data, columns=['Peptide', 'Time', 'Residue', 'Atom', 'Atom_index', 'x', 'y', 'z'])
        del peptide_data  # pour libérer la ram
        print("df créé")

        # Créer une colonne 'Residue_index' contenant les chiffres de la colonne 'Residue'
        df['Residue_index'] = df['Residue'].str.extract('(\d+)')
        print("Residue_index créé")

        # Créer une colonne 'Residue' contenant les noms de résidus (sans les chiffres)
        df['Residue'] = df['Residue'].str.lstrip('0123456789')
        print("Residue mis à jour")

        # Réorganisation des colonnes
        df = df.reindex(columns=['Peptide', 'Time', 'Residue_index', 'Residue', 'Atom', 'Atom_index', 'x', 'y', 'z'])
        print("Réindexage des colonnes terminé")

        # Conversion des colonnes
        df['Residue_index'] = df['Residue_index'].astype(int)
        df['Atom_index'] = df['Atom_index'].astype(int)
        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)
        df['z'] = df['z'].astype(float)
        print("Conversions terminés")

        # Afficher les premières lignes du DataFrame
        print(df.head())

        # Sauvegarde du dataframe
        csv_name = peptide_name + '.csv'
        df.to_csv(csv_name, index=False)
        print("Sauvegarde du csv faite")

        del df  # pour libérer la ram
        print("Ram libérée")
        print("")