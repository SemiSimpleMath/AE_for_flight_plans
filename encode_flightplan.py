import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import pandas as pd
import utils

DATA_PATH = "./data/flight_plans/"
DATA_FILE = DATA_PATH + "plans1.csv"

s = "NEWSC0123456789e"
d = list(s)
plans = utils.generate_flight_plans(100000, d) #100000 is a lot and this step may take couple of minutes to finish

utils.write_plans("./data/flight_plans/plans1.csv", plans)  # This folder needs to exist


def load_data():
    df = pd.read_csv(DATA_FILE)

    import utils
    x_train = []

    for index, row in df.iterrows():
        x = (row['input'])
        x = utils.convert_string_flight_plan_to_torch(x)

        x_train.append(torch.flatten(x))

    return x_train


class Model(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size, decoder_in_size, decoder_out_size):
        super(Model, self).__init__()
        self.encoder = nn.Linear(encoder_in_size, encoder_out_size)
        self.decoder = nn.Linear(decoder_in_size, decoder_out_size)

    def forward(self, x):
        x = F.leaky_relu(self.encoder(x))
        x = F.leaky_relu(self.decoder(x))  # F.relu(self.decoder(x))
        # x = torch.sigmoid(x)
        return x


class Dataset:
    def __init__(self, x): self.x = x

    def __len__(self): return len(self.x)

    def __getitem__(self, i): return self.x[i]


def train(model, opt, num_epochs, dl):
    model.train()

    loss_func = nn.MSELoss()
    counter = 0

    for epoch in range(num_epochs):
        for xb in dl:
            result = model(xb)
            # print(result, xb)
            loss = loss_func(result, xb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            counter += 1
            if counter % 100 == 0:
                print(counter, loss)


def main():
    encoder_in_size = 256
    encoder_out_size = 20
    decoder_in_size = 20
    # decoder_out_size must be same as the encoder_in_size since we are trying to reconstruct the input
    decoder_out_size = encoder_in_size

    model = Model(encoder_in_size, encoder_out_size, decoder_in_size, decoder_out_size)

    print(model)

    # Batch size
    bs = 256

    x_train = load_data()

    train_ds = Dataset(x_train)
    train_dl = DataLoader(train_ds, bs, sampler=RandomSampler(train_ds))

    # Optimizer
    learning_rate = 1e-5
    opt = torch.optim.Adamax(model.parameters(), learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    train(model, opt, 500000, train_dl)


main()
