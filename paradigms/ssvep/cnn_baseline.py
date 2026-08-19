import torch
import torch.nn as nn


class SSVEPNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (n_channels, 1))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, (1, 32), padding=(0, 16))
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def train_cnn(X_train, y_train, X_val, y_val, n_classes, epochs=50, lr=1e-3, device="cpu"):
    model = SSVEPNet(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.long).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.long).to(device)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_acc = (model(Xv).argmax(1) == yv).float().mean().item()
    return model, val_acc
