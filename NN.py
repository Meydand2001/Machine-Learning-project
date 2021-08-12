import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import Read_Data
import Confusion_Matrix


def getdata():
    features, target = Read_Data.read_data3('mushrooms_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20)
    return X_train.values, X_test.values, y_train.values, y_test.values


class NN(nn.Module,):
    def __init__(self,input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20,9)
        self.dropout = nn.Dropout(0.15)


    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

def train(epoch, net, train_x, train_y, optimizer):
    for e in range(epoch):
        optimizer.zero_grad()
        output = net(train_x)
        loss = F.cross_entropy(output, train_y)
        loss.backward()
        optimizer.step()
    return net


def predict(model, test_x):

    pred = []
    with torch.no_grad():
        for data in test_x:
            output = model(data)
            predict = np.argmax(output)
            pred.append(int(predict))

    return pred


def random_state(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    # if you are using GPU
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def main(train_x,test_x,train_y,test_y,input_size):
    random_state(100)
    train_x = train_x
    test_x = test_x
    train_y = train_y
    test_y = test_y
    train_x_tensor = torch.from_numpy(train_x).float()
    train_y_tensor = torch.from_numpy(train_y).long()
    test_x_tensor = torch.from_numpy(test_x).float()
    net = NN(input_size=input_size)
    lr = 0.1
    m = 0.9
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=m)


    my_net = train(100,net,train_x_tensor,train_y_tensor,optimizer)
    predicted = predict(my_net, test_x_tensor)
    actual = test_y
    f1, recall, accuracy = Confusion_Matrix.main(actual, predicted,"Confusion Matrix: Neural Network")
    return f1, recall, accuracy







