import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import time
from utils import device
from pmodel import PolicyModel


class PolicyLearner:
    EPOCH = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 0.0001
    TRAIN_PATH = "policyData/train_data.csv"
    TEST_PATH = "policyData/test_data.csv"

    PRETRAINED_PATH = ""
    PRETRAINED_EPOCH = 0
    SAVE_EVERY = 1
    CHECKPOINTS_FOLDER = "checkpoints/pmodel/"

    def __init__(self):
        self.predictor = PolicyModel()
        if self.PRETRAINED_PATH:
            self.predictor.load_state_dict(torch.load(self.PRETRAINED_PATH, map_location=device))
            print(f'Loaded weights from "{self.PRETRAINED_PATH}".')
        else:
            self.initialize_weights()
        self.predictor.to(device)

        self.loss_fn = nn.CrossEntropyLoss()

        if not os.path.exists(self.CHECKPOINTS_FOLDER):
            os.mkdir(self.CHECKPOINTS_FOLDER)

    def initialize_weights(self):
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def train(self):
        self.predictor.train()
        dataloader = DataLoader(GameDataset(self.TRAIN_PATH), batch_size=self.BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.predictor.parameters(), lr=self.LEARNING_RATE)

        num_batches = len(dataloader)
        for e in range(self.PRETRAINED_EPOCH, self.EPOCH):
            print(f'Epoch: {e + 1} / {self.EPOCH}')
            start_time = time.time()
            for b, (data, label) in enumerate(dataloader):
                pred = self.predictor(data)
                loss = self.loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (b + 1) % 50 == 0:
                    end_time = time.time()
                    print(f"batch: {b+1}/{num_batches}, loss: {loss:.6f}, time: {end_time-start_time:.2f}s")

            # save model
            if (e + 1) % self.SAVE_EVERY == 0:
                _, test_loss = self.evaluate()
                save_path = os.path.join(self.CHECKPOINTS_FOLDER, f"ep{e+1}_lr{self.LEARNING_RATE}_loss{test_loss:.4f}.pth")
                torch.save(self.predictor.state_dict(), save_path)
                print(f'Saved checkpoint to "{save_path}".')
                self.predictor.train()

    def evaluate(self):
        self.predictor.eval()
        dataloader = DataLoader(GameDataset(self.TEST_PATH), batch_size=self.BATCH_SIZE)

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, label in dataloader:
                pred = self.predictor(data)
                test_loss += self.loss_fn(pred, label).item()
                correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
        return correct, test_loss


class GameDataset(Dataset):
    def __init__(self, annotations_file):
        self.dataArray = pd.read_csv(annotations_file, header=None).values

    def __len__(self):
        return self.dataArray.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.dataArray[idx, :], dtype=torch.float32).to(device)
        x = sample[2:].reshape((11, 11))
        if sample[0] == -1:
            y = torch.tensor(121).to(device)
        else:
            y = sample[0] * 11 + sample[1]
        return x, y.long()


if __name__ == "__main__":
    learner = PolicyLearner()
    learner.train()
    learner.evaluate()
