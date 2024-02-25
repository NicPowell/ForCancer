
        
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
# Set print options for tensors
if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False)

    # Load data
    df = pd.read_csv('surveylungcancer.csv')

    df.replace({'YES': 2, 'NO': 1}, inplace=True)


    df.replace({'M': 1, 'F': 2}, inplace=True)


    # Separate features and target
    X = df.drop('LUNG_CANCER', axis=1).values  # Assuming '24' is your target column
    y = df['LUNG_CANCER'].values


    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create data loaders
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    class Net(nn.Module):
        def __init__(self, input_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Initialize model, loss function, and optimizer
    N = X_train.shape[1]  # Number of features
    model = Net(input_size=N)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_model(model, train_loader, criterion, optimizer, epochs=20):
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    def test_model(model, test_loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        print(f'Test Loss: {avg_loss}')
        

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # Test the model
    test_model(model, test_loader, criterion)

    predictions = []
    actual_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.tolist())
            actual_labels.extend(labels.tolist())

    # Convert predictions and actual labels to numpy arrays for easier handling
    predictions = np.array(np.round(predictions))
    actual_labels = np.array(np.round(actual_labels))
    #pd.DataFrame for converting numpy to pd
    predictions = pd.DataFrame(predictions)
    actual_labels = pd.DataFrame(actual_labels)
    predictions.replace({2: 'YES', 1:'NO' }, inplace=True)
    actual_labels.replace({2: 'YES', 1:'NO' }, inplace=True)
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)


    correct_predictions = np.sum(predictions == actual_labels)
    total_predictions = len(actual_labels)
    accuracy = (correct_predictions / total_predictions) * 100
    print(f'{accuracy}%')
    model_folder = 'saved_model'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_path = os.path.join(model_folder, 'model.pth')
    print(model_path)
    torch.save(model.state_dict(), model_path)

