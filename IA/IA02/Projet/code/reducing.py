import os
import csv

def reduce_csv_file(input_file, output_file, reduction_rate):
    """
    Réduit la taille d'un fichier CSV en gardant les lignes où la colonne Time change de valeur reduction_rate fois.

    Args:
        input_file (str): Chemin du fichier CSV d'entrée.
        output_file (str): Chemin du fichier CSV de sortie.
        reduction_rate (int): Nombre de changements de valeur à observer avant de garder une ligne.
    """
    with open(input_file, 'r') as csv_in, open(output_file, 'w', newline='') as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)

        # Lire l'en-tête et identifier l'index de la colonne spécifiée
        header = next(reader)
        writer.writerow(header)
        time_idx = header.index("Time")

        # Initialiser les variables de suivi
        change_count = reduction_rate  # = reduction_rate pour conserver les lignes Time == 0
        last_time_value = '0'

        # Lire et écrire les lignes en fonction du changement de valeur de la colonne 'time'
        for row in reader:
            current_time_value = row[time_idx]
            if current_time_value != last_time_value:
                if change_count == reduction_rate:
                    change_count = 1
                else:
                    change_count += 1
                last_time_value = current_time_value

            # Écrire la ligne si le nombre de changements de valeur atteint le taux de réduction
            if change_count >= reduction_rate:
                writer.writerow(row)

def reduce_csv_file2(input_file, output_file, reduction_max):
    """
    Réduit la taille d'un fichier CSV en gardant les lignes où la colonne Time est inférieur à reduction_max.

    Args:
        input_file (str): Chemin du fichier CSV d'entrée.
        output_file (str): Chemin du fichier CSV de sortie.
        reduction_max (int): Nombre max de changements de valeur à observer pour ne plus conserver de lignes.
    """
    with open(input_file, 'r') as csv_in, open(output_file, 'w', newline='') as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)

        # Lire l'en-tête et identifier l'index de la colonne spécifiée
        header = next(reader)
        writer.writerow(header)
        time_idx = header.index("Time")

        # Lire et écrire les lignes en fonction du changement de valeur de la colonne 'time'
        for row in reader:
            current_time_value = int(row[time_idx])

            # Écrire la ligne si la valeur de Time reste inférieur à reduction_max
            if current_time_value <= reduction_max:
                writer.writerow(row)
            else:
                break

def reduce_csv(files_directory, reduction_rate, reduction_max):
    """Fonction principale du fichier Python"""
    for file_name in os.listdir(files_directory):
        if file_name.endswith('.csv') and not "reduced" in file_name and not "peptides" in file_name:  # pour chaque fichier CSV original
            print(file_name)
            input_file = os.path.join(files_directory, file_name)
            #reduce_csv_file(input_file, os.path.join(files_directory, f"{file_name.split('.')[0]}_reduced.csv"), reduction_rate)
            reduce_csv_file2(input_file, os.path.join(files_directory, f"{file_name.split('.')[0]}_reduced2.csv"), reduction_max)

# Exemple d'utilisation
files_directory = "./"
reduce_csv(files_directory, 10, 20000)