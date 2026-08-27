# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Defining the data loader
class PeptideDataset(Dataset):
    def __init__(self, directory, max_atoms=89, seq_length=4001):
        self.directory = directory
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('reduced2.csv')]
        self.max_atoms = max_atoms
        self.seq_length = seq_length
        self.activity_map = {
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
            'VVGPGA': 0,

            'APGVGV_reduced2': 0,  # inactif
            'EGFEPG_reduced2': 1,  # actif
            'GVAPGV_reduced2': 1,
            'GVGVAP_reduced2': 0,
            'LGTIPG_reduced2': 1,
            'PGAIPG_reduced2': 1,
            'PGAYPG_reduced2': 1,
            'PGVGVA_reduced2': 0,
            'VAPGVG_reduced2': 0,
            'VGLAPG_reduced2': 1,
            'VGVAPG_reduced2': 1,
            'VVGPGA_reduced2': 0
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = pd.read_csv(file_path)
        grouped = data.groupby(['Atom_index'])

        sequences = []
        for _, group in grouped:
            seq = group[['x', 'y', 'z']].values
            if len(seq) < self.seq_length:
                padding = np.zeros((self.seq_length - len(seq), 3))
                seq = np.vstack((seq, padding))
            sequences.append(seq[:self.seq_length])

        sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
        if sequences.shape[0] < self.max_atoms:
            padding = torch.zeros((self.max_atoms - sequences.shape[0], self.seq_length, 3))
            sequences = torch.cat([sequences, padding], dim=0)

        peptide_seq = os.path.basename(file_path).split('.')[0]
        actif = torch.tensor([self.activity_map[peptide_seq]], dtype=torch.float32)

        return sequences, actif

# Utilisation
data_directory = './'
dataset = PeptideDataset(data_directory, max_atoms=89, seq_length=4001)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Defining the generator model
class Generator(nn.Module):
    def __init__(self, num_atoms, input_size=3, hidden_size=30, seq_length=4001, num_layers=1):
        super(Generator, self).__init__()
        self.num_atoms = num_atoms
        self.seq_length = seq_length
        self.grus = nn.ModuleList([
            nn.GRU(input_size + 1, hidden_size, num_layers, batch_first=True) for _ in range(num_atoms)
        ])
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_size, input_size) for _ in range(num_atoms)
        ])

    def forward(self, x, hidden_states, labels):
        labels_expanded = labels.unsqueeze(-1).expand(-1, self.num_atoms, -1)
        x = torch.cat((x, labels_expanded), dim=2)
        outputs = []
        for i, (gru, output_layer) in enumerate(zip(self.grus, self.output_layers)):
            atom_outputs = []
            gru_input = x[:, i:i + 1, :]

            for _ in range(self.seq_length):
                out, hidden_states[i] = gru(gru_input, hidden_states[i])
                out = output_layer(out[:, -1, :])
                atom_outputs.append(out)
                gru_input = torch.cat((out.unsqueeze(1), labels_expanded[:, i:i + 1, :]), dim=2)

            outputs.append(torch.stack(atom_outputs, dim=1))

        return torch.cat(outputs, dim=1), hidden_states

    def init_hidden(self, batch_size, hidden_size):
        return [torch.zeros(1, batch_size, hidden_size) for _ in range(self.num_atoms)]

# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, num_atoms, input_size=3, hidden_size=30, seq_length=4001, num_layers=1):
        super(Discriminator, self).__init__()
        self.num_atoms = num_atoms
        self.seq_length = seq_length
        self.grus = nn.ModuleList([
            nn.GRU(input_size, hidden_size, num_layers, batch_first=True) for _ in range(num_atoms)
        ])
        self.fc_real = nn.Linear(num_atoms * hidden_size, 1)
        self.fc_active = nn.Linear(num_atoms * hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_states):
        #print("Shape of x initially:", x.shape)

        features = []
        new_hidden_states = []
        for i, gru in enumerate(self.grus):
            input_to_gru = x[:, i, :, :]  #  [batch_size, num_atoms, seq_length, features]
            #print(f"Shape of input_to_gru for atom {i}:", input_to_gru.shape)  

            hidden_state_to_gru = hidden_states[i]
            #print(f"Shape of hidden_state_to_gru for atom {i}:", hidden_state_to_gru.shape) 

            out, new_hidden = gru(input_to_gru, hidden_state_to_gru)
            new_hidden_states.append(new_hidden)
            features.append(out[:, -1, :]) 

        features = torch.cat(features, dim=1)
        real_output = self.sigmoid(self.fc_real(features))
        active_output = self.sigmoid(self.fc_active(features))

        return real_output, active_output, new_hidden_states

    def init_hidden(self, batch_size, hidden_size):
        return [torch.zeros(1, batch_size, hidden_size) for _ in range(self.num_atoms)]

# Define the training loop
seq_length = 4001
num_atoms = 89
hidden_size = 30

generator = Generator(num_atoms=num_atoms, seq_length=seq_length)
discriminator = Discriminator(num_atoms=num_atoms, seq_length=seq_length)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Load the states of models or optimizes
if 'ganGRU_checkpoint.pth' in os.listdir(data_directory):
    checkpoint = torch.load('ganGRU_checkpoint.pth')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

else:
    num_epochs = 100
    for epoch in range(num_epochs):
        start_time = time.time()  
        print(f'Starting epoch {epoch + 1}/{num_epochs}')
        for i, (real_data, labels) in enumerate(dataloader):
            batch_size = real_data.size(0)

            # Initialize hidden states
            hidden_gen = generator.init_hidden(batch_size, hidden_size)
            hidden_disc = discriminator.init_hidden(batch_size, hidden_size)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real data
            real_output, real_active_output, _ = discriminator(real_data, hidden_disc)
            loss_real = criterion(real_output, torch.ones(batch_size, 1))
            loss_real_active = criterion(real_active_output, labels.unsqueeze(1).squeeze(-1))  

            # Fake data
            noise = torch.randn(batch_size, num_atoms, 3)
            fake_data, _ = generator(noise, hidden_gen, labels)
            fake_data = fake_data.view(batch_size, num_atoms, seq_length, -1)  # [batch_size, num_atoms, seq_length, features]
            print("Shape of fake_data:", fake_data.shape) 
            fake_output, fake_active_output, _ = discriminator(fake_data.detach(), hidden_disc)
            loss_fake = criterion(fake_output, torch.zeros(batch_size, 1))
            loss_fake_active = criterion(fake_active_output, labels.unsqueeze(1).squeeze(-1))

            # Total loss for discriminator
            loss_D = (loss_real + loss_fake) / 2
            loss_D_active = (loss_real_active + loss_fake_active) / 2
            total_loss_D = loss_D + loss_D_active
            total_loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            fake_output, fake_active_output, _ = discriminator(fake_data, hidden_disc)
            loss_G = criterion(fake_output, torch.ones(batch_size, 1))
            loss_G_active = criterion(fake_active_output, labels.unsqueeze(1).squeeze(-1))
            total_loss_G = loss_G + loss_G_active
            total_loss_G.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss_D: '
                      f'{total_loss_D.item()}, Loss_G: {total_loss_G.item()}')

        end_time = time.time()  
        print(f'Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds')

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, 'ganGRU_checkpoint.pth')

generator.eval()
discriminator.eval()

batch_size = 1  # Nombre de peptides à générer
hidden_size = 30  # Taille des états cachés
num_atoms = 89  # Nombre maximal d'atomes (doit correspondre à la valeur utilisée lors de l'entraînement)
input_size = 3  # Dimension de chaque entrée (x, y, z)
seq_length = 4001

# Générez des peptides actives
# Créez des vecteurs de bruit
noise = torch.randn(batch_size, num_atoms, input_size)

# Spécifiez une étiquette de classe pour la génération (0 pour inactif, 1 pour actif)
label = torch.tensor([1], dtype=torch.float32).unsqueeze(0)

# Initialisez les états cachés du générateur et discriminateur
hidden_gen = generator.init_hidden(batch_size, hidden_size)
hidden_disc = discriminator.init_hidden(batch_size, hidden_size)

# Générez la peptide
with torch.no_grad():  # Pas besoin de calcul des gradients
    generated_peptides, _ = generator(noise, hidden_gen, label)
    generated_peptides = generated_peptides.view(batch_size, num_atoms, seq_length, -1)  

output_dir = "gen_peptide_active"
os.makedirs(output_dir, exist_ok=True)

# Transformer les données générées pour le CSV
for peptide_idx in range(batch_size):
    peptides_list = []
    for atom_index in range(num_atoms):
        for time_step in range(seq_length):
            x, y, z = generated_peptides[peptide_idx, atom_index, time_step].tolist()
            peptides_list.append({
                'Times': time_step * 5,
                'Atom_index': atom_index + 1,
                'x': x,
                'y': y,
                'z': z
            })
    # Créer un DataFrame et sauvegarder dans un fichier CSV
    df = pd.DataFrame(peptides_list)
    df['Atom_index'] = df['Atom_index'].astype(int)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    file_name = os.path.join(output_dir, f"{peptide_idx + 1}.csv")
    df.to_csv(file_name, index=False)

# Classifiez les peptides générées avec le discriminateur
with torch.no_grad():  # Pas besoin de calcul des gradients
    real_output, active_output, _ = discriminator(generated_peptides, hidden_disc)

real_output_list = real_output.squeeze().tolist()
active_output_list = active_output.squeeze().tolist()

# Créer un graphique en ligne
plt.figure(figsize=(10, 6))
plt.plot(range(1, batch_size + 1), real_output_list, label='Probabilité d\'être réel', marker='o')
plt.plot(range(1, batch_size + 1), active_output_list, label='Probabilité d\'être actif', marker='x')
plt.xlabel('Peptide générée')
plt.ylabel('Probabilité')
plt.title('Probabilités de distinction réel/faux et d\'activité pour 10 peptides générées')
plt.legend()
plt.grid(True)

# Sauvegarder le graphique
plt.savefig('peptides_active_gen_probabilities.png')

# Générez maintenant des peptides inactives
# Créez des vecteurs de bruit
noise = torch.randn(batch_size, num_atoms, input_size)

# Spécifiez une étiquette de classe pour la génération (0 pour inactif, 1 pour actif)
label = torch.tensor([0], dtype=torch.float32).unsqueeze(0)

# Initialisez les états cachés du générateur et discriminateur
hidden_gen = generator.init_hidden(batch_size, hidden_size)
hidden_disc = discriminator.init_hidden(batch_size, hidden_size)

# Générez la peptide
with torch.no_grad():  # Pas besoin de calcul des gradients
    generated_peptides, _ = generator(noise, hidden_gen, label)
    generated_peptides = generated_peptides.view(batch_size, num_atoms, seq_length, -1) 

output_dir = "gen_peptide_inactive"
os.makedirs(output_dir, exist_ok=True)

# Transformer les données générées pour le CSV
for peptide_idx in range(batch_size):
    peptides_list = []
    for atom_index in range(num_atoms):
        for time_step in range(seq_length):
            x, y, z = generated_peptides[peptide_idx, atom_index, time_step].tolist()
            peptides_list.append({
                'Times': time_step * 5,
                'Atom_index': atom_index + 1,
                'x': x,
                'y': y,
                'z': z
            })
    # Créer un DataFrame et sauvegarder dans un fichier CSV
    df = pd.DataFrame(peptides_list)
    df['Atom_index'] = df['Atom_index'].astype(int)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    file_name = os.path.join(output_dir, f"{peptide_idx + 1}.csv")
    df.to_csv(file_name, index=False)

# Classifiez les peptides générées avec le discriminateur
with torch.no_grad():  # Pas besoin de calcul des gradients
    real_output, active_output, _ = discriminator(generated_peptides, hidden_disc)

real_output_list = real_output.squeeze().tolist()
active_output_list = active_output.squeeze().tolist()

# Créer un graphique en ligne
plt.figure(figsize=(10, 6))
plt.plot(range(1, batch_size + 1), real_output_list, label='Probabilité d\'être réel', marker='o')
plt.plot(range(1, batch_size + 1), active_output_list, label='Probabilité d\'être actif', marker='x')
plt.xlabel('Peptide générée')
plt.ylabel('Probabilité')
plt.title('Probabilités de distinction réel/faux et d\'activité pour 10 peptides générées')
plt.legend()
plt.grid(True)

# Sauvegarder le graphique
plt.savefig('peptides_inactive_gen_probabilities.png')
