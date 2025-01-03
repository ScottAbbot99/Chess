import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_df():

    CACHE_FILE = "df_cache_1mill.pkl"

    try:
        # Load the cached DataFrame
        with open(CACHE_FILE, "rb") as f:
            df = pickle.load(f)
        print("Loaded data from cache")
    except FileNotFoundError:
        # Load the DataFrame and save it to the cache
        df = pd.read_pickle("stockfish_evaluations_1mill.pkl")
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
        print("Loaded data from local and cached it")
    
    return df


def preprocess_fen(fen):
    fen_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
    }

    # Initialize the board
    board = [0] * 64
    fen_parts = fen.split()[0]  # Only take the board portion of the FEN
    state = fen_parts.replace('/', "")  # Remove row separators

    idx = 0
    for char in state:
        if char.isdigit():
            idx += int(char)  # Skip over empty squares
        else:
            board[idx] = fen_map[char]  # Map the piece to the corresponding number
            idx += 1

    # Extract additional features from FEN
    turn = 1 if fen.split()[1] == 'w' else -1  # 1 for white, -1 for black
    castling_rights = [0] * 4
    castling_str = fen.split()[2]
    castling_rights[0] = 1 if 'K' in castling_str else 0  # White kingside
    castling_rights[1] = 1 if 'Q' in castling_str else 0  # White queenside
    castling_rights[2] = 1 if 'k' in castling_str else 0  # Black kingside
    castling_rights[3] = 1 if 'q' in castling_str else 0  # Black queenside
    en_passant = 0
    en_passant_str = fen.split()[3]
    if en_passant_str != '-':
        en_passant = 1  # En passant possible

    # Return the board and additional features
    return board, turn, castling_rights, en_passant

# Preprocess the FEN data
def preprocess_dataframe(row):
    board, turn, castling_rights, en_passant = preprocess_fen(row)
    return pd.Series({
        'Board': board,
        'Turn': turn,
        'Castling_K': castling_rights[0],
        'Castling_Q': castling_rights[1],
        'Castling_k': castling_rights[2],
        'Castling_q': castling_rights[3],
        'En_Passant': en_passant
    })

def apply_preprocessing(df):
    df[['Board', 'Turn', 'Castling_K', 'Castling_Q', 'Castling_k', 'Castling_q', 'En_Passant']] = df['fen'].apply(preprocess_dataframe)

    return df

def update_evaluation(row):
    """
    Set evaluation to 50 if it's white's turn and mate is available, and -50 if it's black's turn
    Otherwise convert the string value to a float
    """

    evaluation = row['evaluation']
    turn = row['Turn']

    if evaluation.startswith('M'):
        if turn == 'w':
            return 50.0
        else:
            return -50.0
    else:
        return float(evaluation)

def update_dataframe_eval(df):
    df['evaluation'] = df.apply(update_evaluation, axis = 1)

    return df



class ChessDataset(Dataset):
    def __init__(self, data):
        self.board = data["Board"]
        self.turn = data["Turn"]
        self.castling_K = data["Castling_K"]
        self.castling_Q = data["Castling_Q"]
        self.castling_k = data["Castling_k"]
        self.castling_q = data["Castling_q"]
        self.en_passant = data["En_Passant"]
        self.evaluations = data["evaluation"]

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        # Retrieve the row once and unpack it
        board = self.board.iloc[idx]
        turn = self.turn.iloc[idx]
        castling_K = self.castling_K.iloc[idx]
        castling_Q = self.castling_Q.iloc[idx]
        castling_k = self.castling_k.iloc[idx]
        castling_q = self.castling_q.iloc[idx]
        en_passant = self.en_passant.iloc[idx]
        evaluation = self.evaluations.iloc[idx]

        # Combine scalar features into a single tensor
        scalar_features = torch.tensor(
            [turn, castling_K, castling_Q, castling_k, castling_q, en_passant], 
            dtype=torch.float32
        )

        # Convert board and evaluation to tensors
        board = torch.tensor(board, dtype=torch.float32)
        evaluation = torch.tensor(evaluation, dtype=torch.float32)

        # Return the board, combined scalar features, and evaluation
        return board, scalar_features, evaluation


class ChessModel(nn.Module):
    def __init__(self, board_size, scalar_size, hidden_size):
        super(ChessModel, self).__init__()
        # Combine board and scalar features
        self.input_size = board_size + scalar_size
        self.linear1 = nn.Linear(self.input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1)
        self.relu = nn.ReLU()

    def forward(self, board, scalar_features):
        # Concatenate board and scalar features
        x = torch.cat((board, scalar_features), dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

class ChessConvolutionModel(nn.Module):
    def __init__(self):
        super(ChessConvolutionModel, self).__init__()

        # Convolutional layers for the board input
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),  # Input: 1 channel (8x8 board)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),  # Output: 64 channels
            nn.ReLU(),
            nn.Flatten()  # Flatten for fully connected layers
        )

        # Fully connected layers for concatenated features
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 6, 128),  # Combine board and scalar features
            nn.ReLU(),
            nn.Linear(128, 1)  # Final output (e.g., evaluation score or classification)
        )

    def forward(self, board, scalar_features):
        # Pass the board through the convolutional layers
        board_features = self.conv(board)
        # Concatenate board features with scalar features
        x = torch.cat((board_features, scalar_features), dim=1)
        # Pass through fully connected layers
        x = self.fc(x)
        return x




def train(model, train_loader, loss_function, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for board, scalar_features, targets in train_loader:
            # Move tensors to the appropriate device
            board = board.to(device)
            board = board.view(-1, 1, 8, 8)  # Reshape to batch_size x 1 x 8 x 8
            #print(board.shape)  # Expect [batch_size, 1, 8, 8]
            scalar_features = scalar_features.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(board, scalar_features)
            loss = loss_function(outputs, targets.unsqueeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


def evaluate(model, test_loader, loss_function):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for board, scalar_features, targets in test_loader:
            # Move tensors to the appropriate device
            board = board.to(device)
            board = board.view(-1, 1, 8, 8)  # Reshape to batch_size x 1 x 8 x 8
            scalar_features = scalar_features.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(board, scalar_features)
            loss = loss_function(outputs, targets.unsqueeze(1))
            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss}")
        return avg_loss


def initiate_training():

    df = get_df()
    df = apply_preprocessing(df)
    df = update_dataframe_eval(df)

    # Split data and reset indices
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=0)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # create datasets
    train_dataset, test_dataset = ChessDataset(train_data), ChessDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Model dimensions
    """
    board_size = 64  # For the board
    scalar_size = 6  # For scalar features (turn, castling rights, en passant)
    hidden_size = [128, 64]
    """
    
    model = ChessConvolutionModel().to(device)


    # loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training / testing
    for epoch in range(10):
        train(model, train_loader, loss_function, optimizer, num_epochs=1)
        evaluate(model, test_loader, loss_function)

    # save model after training
    torch.save(model.state_dict(), "chess_model_1mil.pth")


if __name__ == "__main__":
    initiate_training()
