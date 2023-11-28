import lightning
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from Dataset import PeopleArtDataset


def collate_fn(data: list[tuple[Tensor, dict[str, Tensor]]]):
    images, annotations = zip(*data)
    return images, annotations


class PeopleArtDataModule(lightning.LightningDataModule):
    annotations_path = "Annotations"

    def __init__(self, data_dir, batch_size=32, workers=8):
        super().__init__()
        self.people_art_predict = None
        self.people_art_test = None
        self.people_art_train = None
        self.people_art_val = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = PeopleArtDataset("PeopleArt")
            self.people_art_train, self.people_art_val = random_split(dataset,
                                                                      [0.9, 0.1],
                                                                      generator=torch.Generator().manual_seed(533))
        if stage == "test":
            dataset = PeopleArtDataset(self.data_dir, mode="test")
            self.people_art_test = dataset
        if stage == "predict":
            dataset = PeopleArtDataset(self.data_dir)
            self.people_art_predict = dataset

    def train_dataloader(self):
        return DataLoader(self.people_art_train, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.people_art_val, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=self.workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.people_art_test, batch_size=self.batch_size, collate_fn=collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.people_art_predict, batch_size=self.batch_size, collate_fn=collate_fn)
