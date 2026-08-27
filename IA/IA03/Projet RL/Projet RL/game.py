import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from collections import namedtuple, deque
from itertools import count
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from debugpy.common.timestamp import current


class ConnectFourEnv:
    # Constructeur de la classe, initialisation de l'environnement avec des dimensions par défaut (6 lignes, 7 colonnes)
    def __init__(self, rows=6, cols=7):
        self.rows = rows  # Définition du nombre de lignes du plateau
        self.cols = cols  # Définition du nombre de colonnes du plateau
        self.board = np.zeros((rows, cols))  # Initialisation du plateau de jeu, rempli de zéros (cases vides)
        self.players = [1, 2]  # Définition des deux joueurs (1 et 2)
        self.current_player = 1
        self.num_moves = 0  # Nombre de coups joués jusqu'à présent
        self.max_moves = rows * cols  # Nombre maximum de coups avant la fin du jeu (plateau rempli)

    # Méthode pour réinitialiser l'environnement (plateau et joueur)
    def reset(self):
        self.board = np.zeros((self.rows, self.cols))  # Initialisation du plateau de jeu, rempli de zéros (cases vides)
        self.num_moves = 0  # Nombre de coups joués jusqu'à présent
        self.current_player = 1
        return self.board

    # Méthode pour obtenir les mouvements valides (colonnes où il reste de la place)
    def get_valid_moves(self):
        return [col for col in range(self.cols) if self.board[self.rows - 1, col] == 0]

    # Méthode qui exécute un coup et met à jour l'état du jeu
    def step(self, action):
        if action not in self.get_valid_moves():
            raise ValueError(f"Invalid action: Column {action} is full or out of range.")

        placed_row = None  # Variable pour stocker la ligne où le jeton est placé
        for row in range(self.rows):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                placed_row = row
                break

        self.num_moves += 1

        # Check if the game has been won
        has_won = self.is_win(placed_row, action)
        is_done = has_won or self.num_moves >= self.max_moves
        if has_won:
            reward = 1
        elif is_done:
            reward = 0.5
        else:
            reward = 0.1
        winner = self.current_player if has_won else None  # Stocke le joueur gagnant si victoire

        # Switch player
        self.current_player = 3 - self.current_player

        return self.board, reward, is_done, winner

    # Méthode pour vérifier si un joueur a gagné après un coup
    def is_win(self, row, col):
        def check_direction(delta_row, delta_col):
            count = 1
            for d in [-1, 1]:
                r, c = row, col
                while True:
                    r += d * delta_row
                    c += d * delta_col
                    if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == self.current_player:
                        count += 1
                    else:
                        break
            return count >= 4

        # Check all directions: horizontal, vertical, and two diagonals
        return (
                check_direction(0, 1) or  # Horizontal
                check_direction(1, 0) or  # Vertical
                check_direction(1, 1) or  # Diagonal
                check_direction(1, -1)  # Diagonal
        )

    # Méthode pour afficher l'état actuel du plateau de jeu
    def render(self):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xticklabels(range(self.cols))
        ax.set_yticklabels(range(self.rows - 1, -1, -1))
        ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.5)

        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = self.board[row, col]
                color = 'white' if cell_value == 0 else 'red' if cell_value == 1 else 'yellow'
                circle = plt.Circle((col, row), 0.4, color=color, ec='black')
                ax.add_patch(circle)

        player1_patch = mpatches.Patch(color='red', label='Player 1')
        player2_patch = mpatches.Patch(color='yellow', label='Player 2')
        ax.legend(handles=[player1_patch, player2_patch], loc='upper right')

        plt.show()

    def set_board(self, board):
        self.board = board

    def set_current_player(self, player):
        self.current_player = player

class RandomAgent:
    def __init__(self, player_id):
        self.player_id = player_id

    def select_action(self, board, valid_moves):
        return random.choice(valid_moves)


# Smarter Agent that selects a winning move, blocks opponent's winning move, or picks a random move
class SmarterAgent:
    def __init__(self, player_id):
        self.player_id = player_id

    def select_action(self, board, valid_moves):
        # Check for a winning move for itself
        for action in valid_moves:
            temp_board = board.copy()
            simulated_row = None
            simulated_env = ConnectFourEnv()
            simulated_env.set_board(temp_board)
            simulated_env.set_current_player(self.player_id)
            for row in range(simulated_env.board.shape[0]):
                if simulated_env.board[row, action] == 0:
                    simulated_env.board[row, action] = self.player_id
                    simulated_row = row
                    break
            if simulated_env.is_win(simulated_row, action):
                print("FOR A WINNING MOVE !!!")
                return action

        # Check for a blocking move against opponent
        for action in valid_moves:
            temp_board = board.copy()
            simulated_row = None
            simulated_env = ConnectFourEnv()
            simulated_env.set_board(temp_board)
            simulated_env.set_current_player(3 - self.player_id)
            for row in range(simulated_env.board.shape[0]):
                if simulated_env.board[row, action] == 0:
                    simulated_env.board[row, action] = 3 - self.player_id
                    simulated_row = row
                    break
            if simulated_env.is_win(simulated_row, action):
                print("FOR A BLOCKING MOVE !!!")
                return action

        # Otherwise, pick a random valid move
        print("FOR A RANDOM MOVE !!!")
        return random.choice(valid_moves)



# Fonction pour compter les fenêtres d'une taille donnée contenant un nombre spécifié de jetons pour un joueur
def count_windows(grid, size, mark, config):
    count = 0
    rows, cols = grid.shape

    # Vérification des directions : horizontal, vertical, diagonales
    for row in range(rows):
        for col in range(cols - 4 + 1):  # Fenêtres horizontales
            window = grid[row, col:col + 4]
            if np.count_nonzero(window == mark) == size and np.count_nonzero(window == 0) == 4 - size:
                count += 1

    for row in range(rows - 4 + 1):
        for col in range(cols):  # Fenêtres verticales
            window = grid[row:row + 4, col]
            if np.count_nonzero(window == mark) == size and np.count_nonzero(window == 0) == 4 - size:
                count += 1

    for row in range(rows - 4 + 1):
        for col in range(cols - 4 + 1):  # Diagonales descendantes
            window = np.array([grid[row + i, col + i] for i in range(4)])
            if np.count_nonzero(window == mark) == size and np.count_nonzero(window == 0) == 4 - size:
                count += 1

        for col in range(4 - 1, cols):  # Diagonales montantes
            window = np.array([grid[row + i, col - i] for i in range(4)])
            if np.count_nonzero(window == mark) == size and np.count_nonzero(window == 0) == 4 - size:
                count += 1

    return count

# Fonction heuristique pour évaluer un coup
def get_heuristic_q1(grid, col, mark, config):
    temp_board = grid.copy()
    for row in range(temp_board.shape[0]):
        if temp_board[row, col] == 0:
            temp_board[row, col] = mark
            break

    A, B, C, D, E = config
    num_fours = count_windows(temp_board, 4, mark, config)
    num_threes = count_windows(temp_board, 3, mark, config)
    num_twos = count_windows(temp_board, 2, mark, config)
    num_threes_opp = count_windows(temp_board, 3, 3 - mark, config)
    num_twos_opp = count_windows(temp_board, 2, 3 - mark, config)

    score = (A * num_fours +
             B * num_threes +
             C * num_twos +
             D * num_twos_opp +
             E * num_threes_opp)
    #print(f"4 : {num_fours}, 3 : {num_threes}, 2 : {num_twos}, 2op : {num_twos_opp}, 3opp : {num_threes_opp}, score : {score}")

    return score

# Agent basé sur une heuristique
class HeuristicAgent:
    def __init__(self, player_id, heuristic_config):
        self.player_id = player_id
        self.heuristic_config = heuristic_config

    def get_action(self, env):
        valid_moves = env.get_valid_moves()
        best_score = float('-inf')
        best_move = None

        for move in valid_moves:
            score = get_heuristic_q1(env.board, move, self.player_id, self.heuristic_config)
            if score > best_score:
                best_score = score
                best_move = move

        if best_score == 0:
            best_move = random.choice(valid_moves)

        return best_move


class MinMaxAgent:
    def __init__(self, player_id, config, depth=3):
        self.player_id = player_id
        self.config = config
        self.depth = depth

    def minimax(self, env, depth, maximizing_player, last_col):
        valid_moves = env.get_valid_moves()

        # Condition d'arrêt : profondeur maximale atteinte ou pas de coup valide
        if depth == 0 or not valid_moves:
            return self.heuristic(env.board, 3-env.current_player, last_col), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None

            for move in valid_moves:
                # Simuler un coup
                temp_env = ConnectFourEnv(env.rows, env.cols)
                temp_env.board = env.board.copy()
                temp_env.current_player = env.current_player
                _, _, _, winner = temp_env.step(move)

                # Appel récursif
                #temp_env.render()
                eval, _ = self.minimax(temp_env, depth - 1, False, move)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            print(f"Depth {depth}, Best move: {best_move}, Score: {max_eval}, current player {env.current_player}")
            return max_eval, best_move

        else:
            min_eval = float('inf')
            best_move = None

            for move in valid_moves:
                # Simuler un coup
                temp_env = ConnectFourEnv(env.rows, env.cols)
                temp_env.board = env.board.copy()
                temp_env.current_player = env.current_player
                _, _, _, winner = temp_env.step(move)

                # Appel récursif
                #temp_env.render()
                eval, _ = self.minimax(temp_env, depth - 1, True, move)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            print(f"Depth {depth}, Best move: {best_move}, Score: {min_eval}, current player {env.current_player}")
            return min_eval, best_move

    def heuristic(self, board, current_player, col):
        # Si col est None, effectuer une évaluation globale
        if col is None:
            return np.sum(board == self.player_id) - np.sum(board == (3 - self.player_id))

        # Utiliser une heuristique basée sur la colonne donnée
        return get_heuristic_q1(board, col, current_player, self.config)

    def get_action(self, env):
        score, action = self.minimax(env, self.depth, True, None)
        print(f"score : {score} --------- action : {action} --------------------------------------------------")
        return action


### RL DQN
# création de la classe Transition (tuple nommé, simplifie les choses)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env = ConnectFourEnv()
env.reset()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# classe de la mémoire tampon pour générer le batch pour l'optimisation du modèle
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Sauvegarde une transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Tirage aléatoire uniforme"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# classe du Deep Q Network
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(n_observations, 8, kernel_size=3, stride=1, padding=1)
        #self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        #self.pool2 = nn.MaxPool2d(2, 2)
        self._conv_output_size = self._get_conv_output_size(n_observations)
        self.fc1 = nn.Linear(self._conv_output_size, int(self._conv_output_size / 2))
        self.fc2 = nn.Linear(int(self._conv_output_size / 2), int(self._conv_output_size / 4))
        self.fc3 = nn.Linear(int(self._conv_output_size / 4), n_actions)

    def _get_conv_output_size(self, n_observations):
        """Méthode pour calculer la taille de sortie après les couches convolutionnelles."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_observations, 6, 7)  # Taille typique pour un plateau 6x7
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            return x.numel()  # Nombre total d'éléments restants

    def forward(self, x):
        # Convolution et pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Aplatir pour les couches entièrement connectées
        x = x.view(x.size(0), -1)

        # Couches entièrement connectées
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# BATCH_SIZE est le nombre de transitions échantillonnées à partir du tampon
# GAMMA est le facteur d'actualisation
# EPS_START est la valeur de départ d'epsilon
# EPS_END est la valeur finale de epsilon
# EPS_DECAY contrôle le taux de décroissance exponentielle d'epsilon, une valeur plus élevée signifie une décroissance plus lente
# TAU est le taux de mise à jour du réseau cible
# LR est le taux d'apprentissage de l'optimiseur AdamW
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Obtient le nombre d'actions possible dans l'environnement (ici 2)
n_actions = int(len(env.get_valid_moves()))
# Obtient le nombre d'observables dans l'environnement (ici 4)
state = env.board
n_observations = 1

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    valid_moves = env.get_valid_moves()  # Liste des indices d'actions valides
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        i = 0
        while True:
            i += 1
            # Calcule Q pour toutes les actions
            with torch.no_grad():
                q_values = policy_net(state.unsqueeze(0).unsqueeze(0))  # Ajoute batch_size et canal : (1, 1, 6, 7)
            action = q_values.max(1).indices.view(1, 1)
            if action in valid_moves:
                return action
            else:
                memory.push(state, action, state, torch.tensor([-0.5]).to(device))
                if i == 10:
                    i = 0
                    optimize_model()
            """
            # Masque les actions invalides (Q = -inf pour actions non valides)
            # Crée un masque d'actions invalides (toutes les actions initialement valides)
            invalid_mask = torch.ones(q_values.size(), device=device, dtype=torch.bool)

            # Marque les actions valides comme non invalides
            invalid_mask[:, valid_moves] = False
            q_values[invalid_mask] = float('-inf')
            # Choisit l'action avec le Q maximum parmi les valides
            return q_values.max(1).indices.view(1, 1)
            """
    else:
        return torch.tensor([[random.choice(valid_moves)]], device=device, dtype=torch.long)


def optimize_model():
    # si pas assez de transitions sauvegardées, pas d'entrainement du modèle
    if len(memory) < BATCH_SIZE:
        return float('+inf')
    # tirage au sort des transitions
    transitions = memory.sample(BATCH_SIZE)
    # convertit en une unique Transition avec des tableaux pour chaque paramètre.
    batch = Transition(*zip(*transitions))

    # Calcule un masque d'états non finaux et concatène les éléments du batch
    # (un état final aurait été celui après lequel la simulation s'est terminée)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s.unsqueeze(0) for s in batch.next_state
                                       if s is not None])
    state_batch = torch.stack([state.unsqueeze(0) for state in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Calcule de Q(s_t, a) - le modèle calcule Q(s_t), puis nous sélectionnons les colonnes des actions entreprises.
    # Il s'agit des actions qui auraient été entreprises pour chaque état du batch selon policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Calcule de V(s_{t+1}) pour tous les états suivants (valeur estimée, ou cible).
    # Les valeurs attendues des actions pour non_final_next_states sont calculées en fonction du target_net
    # en sélectionnant la meilleure récompense avec max(1).values
    # Ceci dépend du masque, de sorte que nous aurons soit la valeur d'état attendue,
    # soit 0 au cas où l'état serait final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    """
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    """
    with torch.no_grad():
        # Sélectionne les meilleures actions dans les états suivants avec policy_net
        next_state_actions = policy_net(non_final_next_states).argmax(1)
        # Estime les valeurs des actions sélectionnées avec target_net
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions.unsqueeze(
            1)).squeeze(1)
    # Calcule des valeurs Q attendues
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Calcul de la perte d'Huber
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimisation du modèle
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()


if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 50



Loss = []


def plot_durations(show_result=False):  # Ne fonctionne pas vraiment
    plt.figure(1)
    loss_t = torch.tensor(Loss, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(loss_t.numpy())
    # On fait la moyenne des 100 derniers épisodes
    if len(loss_t) >= 100:
        means = loss_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if show_result:
        plt.savefig("DQN_training.png")
    plt.show()
    plt.pause(0.001)  # petite pause pour laisser assez de temps de se mettre à jour

def train():
    # Variables pour suivre le meilleur épisode et sauvegarder le modèle
    best_loss = float('+inf')
    nb_win = 0
    for i_episode in range(num_episodes):
        loss = 0
        heuristic_config = (100, 10, 5, -5, -50)  # A, B, C, D, E
        heuristicAgent = HeuristicAgent(player_id=2, heuristic_config=heuristic_config)
        # Initialise l'environnement et récupère son état initial aléatoire
        env.reset()
        state = env.board
        state = torch.tensor(state, dtype=torch.float32, device=device)
        episode_transitions = []  # Stocke les transitions pour cet épisode
        for t in count():
            action = select_action(state)
            observation, reward, terminated, winner = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device)

            if done:
                # Stockage de la transition en mémoire
                memory.push(state, action, next_state, reward)
                episode_transitions.append((state.cpu().numpy(), action.item(), reward.item()))

            # L'état suivant devient l'état actuel
            #state = next_state

            # On optimise le modèle une fois
            loss += float(optimize_model())

            # Mise à jour par interpolation des poids de target_net
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                Loss.append(loss)
                if reward.item() == 1:
                    nb_win += 1
                # Enregistrer le meilleur épisode
                if loss < best_loss:
                    best_loss = loss
                    torch.save(policy_net.state_dict(), "best_model.pth")
                    print(f"New best loss: {best_loss}. Model saved.")
                plot_durations()
                break

            else:  # joueur adverse
                action_adv = heuristicAgent.get_action(env)
                #action_adv = select_action(state).item()
                observation, reward, terminated, winner = env.step(action_adv)
                if reward == 1:
                    reward = -reward  # pour éviter que le draw à 0.5 devienne -0.5
                reward = torch.tensor([reward], device=device)
                done = terminated

                if terminated:
                    next_next_state = None
                else:
                    next_next_state = torch.tensor(observation, dtype=torch.float32, device=device)

                # Stockage de la transition en mémoire
                memory.push(state, action, next_next_state, reward)
                episode_transitions.append((state.cpu().numpy(), action.item(), reward.item()))

                # L'état suivant devient l'état actuel
                state = next_next_state

                if done:
                    Loss.append(loss)
                    # Enregistrer le meilleur épisode
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(policy_net.state_dict(), "best_model.pth")
                        print(f"New best loss: {best_loss}. Model saved.")
                    plot_durations()
                    break


    print(f"nb_win : {nb_win}")
    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


class DQNAgent:
    def __init__(self, player_id, policy_net):
        self.player_id = player_id
        self.policy_net = policy_net

    def select_action(self, board, valid_moves):
        state = torch.tensor(board, dtype=torch.float32, device=device)
        with torch.no_grad():
            while True:
                # Calcule Q pour toutes les actions
                q_values = policy_net(state.unsqueeze(0).unsqueeze(0))
                action = q_values.max(1).indices.view(1, 1)
                if action in valid_moves:
                    return action

                # Masque les actions invalides (Q = -inf pour actions non valides)
                invalid_mask = torch.ones(q_values.size(), device=device, dtype=torch.bool)
    
                # Marque les actions valides comme non invalides
                invalid_mask[:, valid_moves] = False
                q_values[invalid_mask] = float('-inf')
                # Choisit l'action avec le Q maximum parmi les valides
                return q_values.max(1).indices.view(1, 1)




# Fonction pour jouer une partie complète entre deux joueurs
def play_game(agent1, agent2):
    env = ConnectFourEnv()
    env.reset()
    done = False

    while not done:
        current_agent = agent1 if env.current_player == 1 else agent2
        valid_moves = env.get_valid_moves()

        # Obtenir l'action du joueur
        if isinstance(current_agent, RandomAgent) or isinstance(current_agent, SmarterAgent) or isinstance(current_agent, DQNAgent):  # Agent joue
            action = current_agent.select_action(env.board, valid_moves)
            print(f"Player {env.current_player} (Agent) chooses column {action}")
        elif isinstance(current_agent, HeuristicAgent) or isinstance(current_agent, MinMaxAgent):
            action = current_agent.get_action(env)
            print(f"Player {env.current_player} (Agent) chooses column {action}")
        else:  # Joueur humain joue
            print(f"Valid moves: {valid_moves}")
            try:
                action = int(input(f"Player {env.current_player}, choose a column: "))
                if action not in valid_moves:
                    raise ValueError("Invalid move. Try again.")
            except ValueError as e:
                print(e)
                continue

        # Appliquer l'action
        _, reward, done, winner = env.step(action)

        # Afficher le plateau
        env.render()

        # Vérifier le résultat
        if reward > 0.5:
            print(f"Player {winner} wins!")
        elif done:
            print("It's a draw!")


# Configuration des coefficients de l'heuristique
heuristic_config = (100, 10, 5, -5, -50)  # A, B, C, D, E
config = (100, 10, 1, -8, -80)

#print(count_windows(np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]), 3, 1, heuristic_config))
#"""
agent1 = RandomAgent(player_id=2)  # Agent joue comme joueur 1
agent2 = SmarterAgent(player_id=2)  # Agent joue comme joueur 2
agent3 = HeuristicAgent(player_id=2, heuristic_config=heuristic_config)
agent4 = MinMaxAgent(player_id=2, depth=1, config=heuristic_config)
#play_game(agent3, agent4)  # Joueur humain contre agent
#"""

#train()
policy_net_agent = DQN(1, 7)
policy_net_agent.load_state_dict(torch.load("best_model.pth", weights_only=True))
policy_net_agent.eval()
agent5 = DQNAgent(player_id=1, policy_net=policy_net_agent)
play_game(agent5, agent1)