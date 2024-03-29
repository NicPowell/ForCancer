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
    model_path = '/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/saved_model/model.pth'
    # Load the model from the different folder
    mapping = {'yes': 2, 'no': 1}

    # Modify the mapping dictionary to handle only the binary values
    for key, value in data.items():
        if value in mapping:
            mapping[key] = mapping[value]
    
    # Convert the dictionary to a NumPy array
    data = np.array([mapping.get(key, value) for key, value in data.items()])
    
    data = data.astype(int)
    #
    # print(data)
    N = len(data)
    
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
    
    data = torch.tensor(data, dtype=torch.float32)
    output = loaded_model(data)
    output1 = np.round(output.tolist())
    output = output.item()
    predictions = pd.DataFrame(output1)
    predictions.replace({2: 'Positive', 1: 'Negative', 0: 'Negative' }, inplace=True)
    
    if predictions.iloc[0, 0] == 'Positive':
        probability = 2/output
        if probability >=1:
            probability -= 1
    else:
        probability = 1/output
        if probability >=1:
            probability -= 1
    result_sentence = (f"You are statistically likely to be: {predictions.iloc[0, 0]} for lung cancer")

    return result_sentence


    # Test the loaded model

    
    
    #return label

#data = {'question1': 'yes','question2': 74, 'question3': 'no', 'question4': 'yes', 'question5': 'yes', 'question6': 'no', 'question7': 'no', 'question8': 'yes', 'question9': 'no', 'question10': 'yes', 'question11': 'yes', 'question12': 'yes', 'question13': 'yes', 'question14': 'yes', 'question15': 'yes'}
#data = {'question1': 'yes', 'question3': 'yes', 'question4': 'yes', 'question5': 'yes', 'question6': 'yes', 'question7': 'yes', 'question8': 'yes', 'question9': 'yes', 'question10': 'yes', 'question11': 'yes', 'question12': 'yes', 'question13': 'yes', 'question14': 'yes', 'question15': 'yes'}
#data = {'question1': 'no', 'question2': 59, 'question3': 'no', 'question4': 'no', 'question5': 'no', 'question6': 'yes', 'question7': 'no', 'question8': 'yes', 'question9': 'no', 'question10': 'yes', 'question11': 'no', 'question12': 'yes', 'question13': 'yes', 'question14': 'no', 'question15': 'yes'}
#current model trained already seems to work on most everything
# Replace values in the dictionary


if __name__ == "__main__":
    data = {'1': 'yes', '2': '54', '3': 'yes', '4': 'yes', '5': 'yes', '6': 'yes', '7': 'yes', '8': 'yes', '9': 'yes', '10': 'yes', '11': 'yes', '12': 'yes', '13': 'yes', '14': 'yes', '15': 
'yes'}
    predict(data)
