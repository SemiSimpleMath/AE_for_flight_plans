
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# CONSTANTS #

# This is the dimension of the bottle neck.  It is the most important parameter in AE
bottle_neck_size = 256
DATA_PATH = "./data/flight_plans/"
DATA_FILE = DATA_PATH + "plans1.csv"


def load_data():
    df = pd.read_csv(DATA_FILE)
    # x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid))
    # return x_train, y_train, x_valid, y_valid
    import utils
    x_train = []
    y_train = []
    for index, row in df.iterrows():
        x,y = (row['input'], row['target'])
        x = utils.convert_string_flight_plan_to_torch(x)
        y = utils.convert_string_flight_plan_to_torch(y)
        x = x/x.sum()
        x_train.append(x)
        y_train.append(x)

    x_train = torch.stack(x_train,0)
    y_train = torch.stack(y_train, 0)
    print(x_train.shape)
    return x_train, y_train

class Model(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size, decoder_in_size, decoder_out_size):
        super(Model, self).__init__()
        self.encoder = nn.Linear(encoder_in_size, encoder_out_size)
        self.decoder = nn.Linear(decoder_in_size, decoder_out_size)

    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = F.relu(self.decoder(x))
        #x = torch.sigmoid(x)
        return x


class Dataset:
    def __init__(self, x, y): self.x, self.y = x, y

    def __len__(self): return len(self.x)

    def __getitem__(self, i): return self.x[i], self.y[i]




def train(model, opt, dl, num_epochs, bs):
    model.train()

    loss_func = nn.MSELoss()
    counter = 0
    for epoch in range(num_epochs):
        for xb, yb in dl:
            result = model(xb)
            #print(result, xb)
            loss = loss_func(result, xb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            counter += 1
            if counter % 1000 == 0:
                print(counter, loss)



def main():
    x_train, y_train = load_data()


    print(x_train.shape)
    print(y_train.shape)

    # Set the dimensions of the encoder / decoder
    encoder_in_size = x_train.shape[1]
    encoder_out_size = bottle_neck_size
    decoder_in_size = bottle_neck_size
    # decoder_out_size must be same as the encoder_in_size since we are trying to reconstruct the input
    decoder_out_size = encoder_in_size

    model = Model(encoder_in_size, encoder_out_size, decoder_in_size, decoder_out_size)

    print(model)

    train_ds = Dataset(x_train, y_train)

    # Batch size
    bs = 1

    # Dataloader
    train_dl = DataLoader(train_ds, bs, sampler=RandomSampler(train_ds))

    # Optimizer
    learning_rate = 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Epochs
    epochs = 100000

    train(model, opt, train_dl, epochs, bs)




main()
