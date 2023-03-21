import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class GlossModel(nn.Module):
    def __init__(self, input_size: int, class_no: int):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.lstm1 = nn.LSTM(
            input_size, 128, device=self.device, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, device=self.device, batch_first=True)
        self.fc1 = nn.Linear(64, 32, device=self.device)
        self.fc2 = nn.Linear(32, class_no, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm1_out = self.relu(self.lstm1(x)[0])
        lstm2_out = self.relu(self.lstm2(lstm1_out)[0])
        fc1_out = self.relu(self.fc1(lstm2_out))
        out = self.softmax(self.fc2(fc1_out))
        return out
