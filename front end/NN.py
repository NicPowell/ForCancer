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



def predict(data):
    model_path = 'saved_model\model.pth'
    # Load the model from the different folder

    N = data.shape[1]
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
    loaded_model = Net(input_size=N)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    output = loaded_model(X_test)
    print(output)
    return output


    # Test the loaded model

    
    
    #return label

data = {'question1': 'yes', 'question2': 'yes', 'question3': 'yes', 'question4': 'yes', 'question5': 'yes', 'question6': 'yes', 'question7': 'yes', 'question8': 'yes', 'question9': 'yes', 'question10': 'yes', 'question11': 'yes', 'question12': 'yes', 'question13': 'yes', 'question14': 'yes', 'question15': 'yes', 'question16': 'yes'}
mapping = {'yes': 2, 'no': 1}

# Replace values in the NumPy array
for key, value in mapping.items():
    data[data == key] = value
predict(data)
