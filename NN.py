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
from Model_trained import Net
from Model_trained import test_model, test_loader, criterion, X_test
model_path = 'saved_model\model.pth'
# Load the model from the different folder
N = X_test.shape[1]
loaded_model = Net(input_size=N)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()



# Test the loaded model
test_model(loaded_model, test_loader, criterion)

