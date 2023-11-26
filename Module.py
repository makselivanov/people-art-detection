import lightning
import torch
import torchvision
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large


class PeopleArtModule(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.ssd = ssdlite320_mobilenet_v3_large(weights=self.weights, )
        self.model = self.weights.transforms()
        self.metrics = MeanAveragePrecision()

    def forward(self, images, targets=None) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
        return self.ssd.forward(list(map(self.model, images)), targets)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        ssd_metrics = self(inputs, target)
        bbox_regression = ssd_metrics["bbox_regression"]
        return bbox_regression

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        metric = self.calculate_metrics(output, target)

        # log 6 example images
        # or generated text... or whatever
        sample_imgs = inputs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('example_images', grid, 0)

        # calculate acc
        labels_hat = torch.argmax(output, dim=1)
        val_acc = torch.sum(target == labels_hat).item() / (len(target) * 1.0)

        # log the outputs!
        self.log_dict({'val_loss': metric, 'val_acc': val_acc})

    def calculate_metrics(self, output, target):
        self.metrics.update(output, target)
        return self.metrics.compute()

    def configure_optimizers(self):
        return torch.optim.SGD(self.ssd.parameters(), lr=0.1)


if __name__ == '__main__':
    from DataModule import PeopleArtDataModule

    dataloader = PeopleArtDataModule("PeopleArt", batch_size=8)
    model = PeopleArtModule()

    trainer = lightning.Trainer(fast_dev_run=10)
    trainer.fit(model=model, train_dataloaders=dataloader)
