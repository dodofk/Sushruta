from typing import Any, List
from omegaconf import DictConfig
import timm
import torch
import torch.nn as nn
from src.models.basic_module import BaseClassificationModele
from src.models.components.tcn import TemporalConvNet as TCN
from torchmetrics import F1Score
from torchmetrics.classification import ConfusionMatrix


# todo: add statscore to compute when validation
# therefore could easily draw the coffusion matrix
class ResnetTSModule(BaseClassificationModele):
    def __init__(
        self,
        temporal_cfg: DictConfig,
        mlp: DictConfig,
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

        if temporal_cfg.type in ["LSTM", "GRU", "RNN"]:
            self.temporal_model = getattr(nn, temporal_cfg.type)(
                input_size=self.feature_extractor.num_features,
                hidden_size=temporal_cfg.hidden_size,
                num_layers=temporal_cfg.num_layers,
                bidirectional=temporal_cfg.bidirectional,
                batch_first=True,
            )
            self.mlp = nn.Sequential(
                nn.Linear(
                    temporal_cfg.hidden_size * self.temporal_direction(),
                    self.num_class(),
                ),
                # should permute if want to use batch norm
                # nn.BatchNorm1d(mlp.hidden_size),
                # nn.ReLU(),
                # nn.Linear(mlp.hidden_size, self.num_class()),
            )
        elif temporal_cfg.type in ["TCN"]:
            self.model_conv_fc = nn.Linear(
                self.feature_extractor.num_features,
                temporal_cfg.spatial_feat_dim,
            )
            channel_sizes = [temporal_cfg.n_hid] * temporal_cfg.levels
            self.temporal_model = TCN(
                num_inputs=temporal_cfg.spatial_feat_dim,
                num_channels=channel_sizes,
                kernel_size=temporal_cfg.kernel_size,
                dropout=temporal_cfg.dropout,
            )
            self.mlp = nn.Sequential(
                nn.Linear(
                    channel_sizes[-1],
                    self.num_class(),
                ),
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

    def temporal_direction(self):
        if (
            self.hparams.temporal_cfg.type is None
            or not self.hparams.temporal_cfg.bidirectional
        ):
            return 1
        else:
            return 2

    # didn't fine a proper way to solve the prolem, could only training on cuda now
    # todo: solve deivce error
    def frames_feature_extractor(
            self,
            x: torch.Tensor,
            output: torch.Tensor,
    ):
        # output = torch.zeros([x.shape[0], x.shape[1], self.feature_extractor.num_features])
        for i in range(0, x.shape[1]):
            output[:, i, :] = self.feature_extractor(x[:, i, :, :, :])
        return output.to("cuda")

    def forward(self, x):
        output_tensor = torch.zeros([x.shape[0], x.shape[1], self.feature_extractor.num_features])
        x = self.frames_feature_extractor(x, output_tensor)

        if self.hparams.temporal_cfg.type in ["LSTM", "GRU", "RNN"]:
            x, _ = self.temporal_model(x)
            x = x[:, -1, :]
        elif self.hparams.temporal_cfg.type in ["TCN"]:
            x = self.model_conv_fc(x)
            x = x.transpose(1, 2)
            x = self.temporal_model(x)
            x = x.transpose(1, 2)
            x = x[:, -1, :]
        
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
        # TODO: finish the step part and choose the proper loss function for multi-classification
        logits = self.forward(batch["image"])
        loss = self.criterion(logits, batch[self.hparams.task])
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, batch[self.hparams.task]

    def training_step(self, batch: Any, batch_idx: int):
        return super().training_step(batch, batch_idx)

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
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
