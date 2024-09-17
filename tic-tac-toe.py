import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset
# Assuming your data is in 'tic_tac_toe.csv'
data = pd.read_csv('tic-tac-toe.csv')

# Step 2: Preprocess the data
# Convert the board state features (categorical values) into numeric labels
encoder = LabelEncoder()

# Apply the encoder to each column (for board squares)
for col in data.columns[:-1]:  # Exclude the 'Class' column
    data[col] = encoder.fit_transform(data[col])

# Encode the Class column (assuming "positive" and "negative" are the only classes)
data['Class'] = encoder.fit_transform(data['Class'])

# Step 3: Separate features and labels
X = data.iloc[:, :-1]  # All columns except the last (which is 'Class')
y = data['Class']  # The 'Class' column

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Decision Tree Classifier (or use RandomForestClassifier for better results)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Optional: Try Random Forest Classifier if you'd like to experiment

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Function to check for a win in Tic-Tac-Toe
def check_win(board, player):
    win_states = [(0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
                  (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
                  (0, 4, 8), (2, 4, 6)]             # Diagonals
    for state in win_states:
        if all([board[i] == player for i in state]):
            return True
    return False




# Function to display the board
def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    for i in range(0, 9, 3):
        print(f"{symbols[board[i]]} | {symbols[board[i + 1]]} | {symbols[board[i + 2]]}")
        if i < 6:
            print('--|---|--')

# Let the bot play
def bot_move(board):
    available_moves = [i for i in range(9) if board[i] == 0]
    if available_moves:
        prediction = clf.predict([board])[0]
        if prediction in available_moves:
            return prediction
        else:
            return np.random.choice(available_moves)  # Fall back if prediction is invalid
    return None

# Simulate a game with the bot
def play_game():
    board = [0] * 9  # Empty board
    for turn in range(9):
        print(f"\nTurn {turn + 1}:")
        player = 1 if turn % 2 == 0 else -1
        
        if player == 1:  # Bot plays as 'X'
            move = bot_move(board)
            print(f"Bot chooses move {move}")
        else:  # Random opponent (could be replaced by another bot or player input)
            # available_moves = [i for i in range(9) if board[i] == 0]
            move = int(input("Where do you want to go next?"))
            print(f"Opponent chooses move {move}")
        
        board[move] = player
        print_board(board)
        
        if check_win(board, player):
            print(f"\nPlayer {'X' if player == 1 else 'O'} wins!")
            return
    print("\nIt's a draw!")

# Play a game with the trained bot
play_game()
