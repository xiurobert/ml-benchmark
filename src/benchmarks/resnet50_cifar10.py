import torch
import torchvision
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

import argparse


def main(epochs=4):
    train_ds = torchvision.datasets.CIFAR10(root=os.path.abspath(os.path.join("..", "..", "data")),
                                            train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_ds = torchvision.datasets.CIFAR10(root=os.path.abspath(os.path.join("..", "..", "data")),
                                           train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    torch.random.manual_seed(0)
    print("Train dataset size:", len(train_ds))
    print("Test dataset size:", len(test_ds))
    print(f"Batch size: {BATCH_SIZE}")

    model = torchvision.models.resnet50()
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs):
        t0 = time.time()
        for batch_num, (X, y) in (pbar := tqdm(enumerate(train_dl), total=len(train_dl))):
            inputs, targets = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_val = loss(outputs, targets)
            pbar.set_description(f"Loss: {loss_val.item():.4f}")
            loss_val.backward()
            optimizer.step()

        t1 = time.time()

        print(f"Epoch: {epoch+1}, Training time: {t1 - t0:.2f}s")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device_num", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    main(epochs=args.epochs)
