import torch
import torch.nn as nn
import torch.nn.functional as F


# run1 : dropout=0.1, lr=0.001
# run2 : dropout=0.2, lr=0.0005
# run2 : dropout=0.2, lr=0.0001
# run2 : dropout=0.1, lr=0.005
# run3 : dropout=0.1, lr=0.0005, discount_factor=0.99 (25.05.-17:29)
# run3 : dropout=0.1 (l2), lr=0.0005, discount_factor=0.97 (25.05. )


# create model class for nn module
class Model(nn.Module):
    # input layer:
    def __init__(self, output, h1=128, h2=64):
        super().__init__()
        # self.conv = nn.Sequential(
        # nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(inplace=True),
        # nn.Conv2d(32, 64, 4, stride=2),    nn.ReLU(inplace=True),
        # nn.Conv2d(64, 64, 3, stride=1),    nn.ReLU(inplace=True),

        # nn.Linear(3136, 512), nn.ReLU(inplace=True),
        # nn.Linear(512, output))

        self.fc1 = nn.Linear(2, h1)
        # self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(p=0.1)
        self.out = nn.Linear(h2, output)

    # forward path
    def fw(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)

        return x
        # return self.conv(x/255.0)


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier (Glorot) uniform initialization, good for relu
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
