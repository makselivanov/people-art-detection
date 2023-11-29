import lightning
import torch
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large


class PeopleArtModule(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.ssd = ssdlite320_mobilenet_v3_large(weights=self.weights)
        self.model = self.weights.transforms()
        self.metrics = MeanAveragePrecision() # max_detection_thresholds=[1, 5, 50]
        self.metrics.warn_on_many_detections = False

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
        self.log("mAP", metric["map"], batch_size=len(batch), prog_bar=True,  on_step=False, on_epoch=True)
        self.log_dict({key : metric[key].mean() for key in metric if key != 'classes'}, batch_size=len(batch))

    def calculate_metrics(self, output, target):
        self.metrics.update(output, target)
        return self.metrics.compute()

    def configure_optimizers(self):
        return torch.optim.SGD(self.ssd.parameters(), lr=0.1)


if __name__ == '__main__':
    from DataModule import PeopleArtDataModule

    dataloader = PeopleArtDataModule("PeopleArt", batch_size=2)
    model = PeopleArtModule()

    trainer = lightning.Trainer(fast_dev_run=5, default_root_dir="./checkpoints")
    trainer.fit(model=model, train_dataloaders=dataloader)
