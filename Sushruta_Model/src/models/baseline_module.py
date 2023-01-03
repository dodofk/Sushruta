from typing import Any, List
from omegaconf import DictConfig
import timm
import torch
import torch.nn as nn
from src.models.basic_module import BaseClassificationModele
from torchmetrics import F1Score
from torchmetrics.classification import ConfusionMatrix


# todo: rewrite the optim lr weight_decay config in same layer and also lr scheduler
class BaselineModule(BaseClassificationModele):
    def __init__(
            self,
            mlp: DictConfig,
            # temporal_model: DictConfig,
            optim: str = "Adam",
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            task: str = "tool",
            use_timm: bool = False,
            backbone_model: str = "resnet34",
    ):
        super().__init__()

        """
        First implement two model which could handle tool detection or action detection, and could be choose 
        by basic.yaml config about task (self.hparams.task)
        """

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_f1 = F1Score(
            num_classes=self.num_class(),
            average="none",
        )
        self.val_f1 = F1Score(
            num_classes=self.num_class(),
            average="none",
        )
        self.train_f1_macro = F1Score(
            num_classes=self.num_class(),
            average="micro",
        )
        self.val_f1_macro = F1Score(
            num_classes=self.num_class(),
            average="micro",
        )
        self.val_confusion_matrix = ConfusionMatrix(
            num_classes=self.num_class(),
        )

        self.feature_extractor = timm.create_model(
            backbone_model,
            pretrained=use_timm,
            in_chans=3,
            num_classes=0,
        )

        # self.temporal_model = instantiate(temporal_model)

        self.mlp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                self.feature_extractor.num_features,
                self.num_class(),
            ),
            # nn.BatchNorm1d(mlp.hidden_size),
            # nn.ReLU(),
            # nn.Linear(mlp.hidden_size, self.num_class()),
        )

        if task in ["phase"]:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif task in ["tool", "action"]:
            self.criterion = torch.nn.BCEWithLogitsLoss()

    def num_class(self):
        task_class = {
            "tool": 7,
            "phase": 7,
            "action": 4,
        }
        return task_class.get(self.hparams.task)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.mlp(x)
        return x

    def step(self, batch: Any):
        """
        batch would a be a dict might contains the following things
        *image*: the frame image
        *action*: the action [Action type 0, Action type 1, Action type 3, Action type 4]
        *tool*: the tool [Tool 0, Tool 1, ..., Tool 6]
        *phase*: the phase [phase 0, ..., phase 6]

        ex:
        image = batch["image"]
        self.forward(image)

        return

        loss: the loss by the loss_fn
        preds: the pred by our model (i guess it would be sth like preds = torch.argmax(logits, dim=-1))
        y: correspond to the task it should be action or tool
        """
        logits = self.forward(batch["image"])
        loss = self.criterion(logits, batch[self.hparams.task])
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, batch[self.hparams.task]

    def training_step(self, batch: Any, batch_idx: int):
        return super().training_step(batch, batch_idx)

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        return super().validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        super().validation_epoch_end(outputs)

    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        super().on_epoch_end()

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.optim)(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # optimizer = getattr(torch.optim, self.hparams.optim)(
        #     params=self.parameters(),
        #     lr=self.hparams.lr,
        #     weight_decay=self.hparams.weight_decay,
        # )
        # lr_scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler)(
        #     factor=self.hparams.factor
        # )
        # return [optimizer], [lr_scheduler]
