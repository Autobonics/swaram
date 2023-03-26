from torch.utils.data import Dataset
from gloss_proc import GlossProcess


class GlossDataset(Dataset):
    def __init__(self):
        self.gp = GlossProcess.load_checkpoint()
        self.gdata, self.labels = self.gp.get_all_gdata()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.gdata[index], self.labels[index]
