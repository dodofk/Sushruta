import os
from typing import Any, List
from omegaconf import DictConfig
import timm
import torch
import torch.nn as nn
import torch.nn.functional as f
from pytorch_lightning import LightningModule
from torchmetrics import Precision
import ivtmetrics
from hydra.utils import get_original_cwd
import numpy as np
from pprint import pprint


class TripletAttentionModule(LightningModule):
    def __init__(
        self,
        temporal_cfg: DictConfig,
        optim: DictConfig,
        loss_weight: DictConfig,
        tool_component: DictConfig,
        target_tool_attention: DictConfig,
        use_pretrained: bool = True,
        emb_dim: int = 256,
        backbone_model: str = "",
        backbone_trainable: bool = True,
        triplet_map: str = "./data/CholecT45/dict/maps.txt",
        use_pos_weight: bool = True,
        pos_weight_dir: str = "./data/pos_weight",
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_recog_metric = ivtmetrics.Recognition(num_class=100)
        self.valid_recog_metric = ivtmetrics.Recognition(num_class=100)
        self.test_recog_metric = ivtmetrics.Recognition(num_class=100)

        self.class_num = {
            "tool": 6,
            "verb": 10,
            "target": 15,
            "triplet": 100,
        }
        self.train_tool_map = Precision(
            num_classes=self.class_num["tool"],
            average="macro",
        )
        self.train_verb_map = Precision(
            num_classes=self.class_num["verb"],
            average="macro",
        )
        self.train_target_map = Precision(
            num_classes=self.class_num["target"],
            average="macro",
        )
        self.train_triplet_map = Precision(
            num_classes=self.class_num["triplet"],
            average="macro",
        )
        self.valid_tool_map = Precision(
            num_classes=self.class_num["tool"],
            average="macro",
        )
        self.valid_verb_map = Precision(
            num_classes=self.class_num["verb"],
            average="macro",
        )
        self.valid_target_map = Precision(
            num_classes=self.class_num["target"],
            average="macro",
        )
        self.valid_triplet_map = Precision(
            num_classes=self.class_num["triplet"],
            average="macro",
        )

        assert (
            "vit" in backbone_model or "swin" in backbone_model
        ), "Only support using vision transformer based model"

        self.feature_extractor = timm.create_model(
            backbone_model,
            pretrained=use_pretrained,
            in_chans=3,
            num_classes=0,
        )

        for p in self.feature_extractor.parameters():
            p.requires_grad = backbone_trainable

        # swin transformer specific
        if not backbone_trainable:
            for p in self.feature_extractor.layers[-1].parameters():
                p.requires_grad = True

        self.tool_information = nn.Sequential(
            nn.Linear(
                self.feature_extractor.num_features,
                emb_dim,
            ),
            nn.Dropout(p=tool_component.dropout_ratio),
        )

        self.tool_head = nn.Sequential(
            nn.Linear(
                emb_dim,
                self.class_num["tool"],
            ),
        )

        self.attention_pre_fc = nn.Linear(
            self.feature_extractor.num_features,
            emb_dim,
        )

        self.target_tool_attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            batch_first=True,
            **target_tool_attention,
        )

        self.target_head = nn.Sequential(
            nn.Linear(
                emb_dim,
                self.class_num["target"],
            ),
        )

        self.ts = nn.Sequential(
            getattr(nn, temporal_cfg.type)(
                input_size=self.feature_extractor.num_features,
                hidden_size=temporal_cfg.hidden_size,
                num_layers=temporal_cfg.num_layers,
                bidirectional=temporal_cfg.bidirectional,
                batch_first=True,
            ),
        )

        self.ts_fc = nn.Linear(
            temporal_cfg.hidden_size * self.temporal_direction(),
            emb_dim,
        )

        self.verb_head = nn.Sequential(
            nn.Linear(
                emb_dim,
                self.class_num["verb"],
            ),
        )

        self.triplet_head = nn.Sequential(
            nn.Linear(
                emb_dim,
                self.class_num["triplet"],
            ),
        )

        self.triplet_pos_weight = self.contruct_pos_weight(component="triplet")
        self.tool_pos_weight = self.contruct_pos_weight(component="tool")
        self.verb_pos_weight = self.contruct_pos_weight(component="verb")
        self.target_pos_weight = self.contruct_pos_weight(component="target")

        self.tool_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.tool_pos_weight,
        )
        self.verb_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.verb_pos_weight
        )
        self.target_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.target_pos_weight
        )
        self.triplet_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.triplet_pos_weight
        )

        self.triplet_map = self.contstruct_triplet_map()

        self.vit_dim = self.test_dim()

    def test_dim(self):
        self.feature_extractor.eval()
        x = torch.randn(1, 3, 224, 224)
        return self.feature_extractor.forward_features(x).shape[1]

    def contruct_pos_weight(self, component: str = "triplet"):
        assert component in ["tool", "verb", "target", "triplet"]

        if not self.hparams.use_pos_weight:
            return torch.ones(self.class_num[component])

        with open(
            os.path.join(
                get_original_cwd(),
                self.hparams.pos_weight_dir,
                f"{component}_pos_weight.txt",
            ),
            "r",
        ) as f:
            pos_weight = [int(pos) for pos in f.read().split(",")]

        weight_sum = sum(pos_weight)

        cal_weight = [0.0] * len(pos_weight)

        for index, pos in enumerate(pos_weight):
            cal_weight[index] = (weight_sum - pos) / (pos + 1e-5)

        return torch.Tensor(cal_weight)

    def contstruct_triplet_map(self):
        with open(os.path.join(get_original_cwd(), self.hparams.triplet_map), "r") as f:
            triplet_map = f.read().split("\n")[1:-2]

        ret = list()
        for triplet in triplet_map:
            ret.append(list(map(int, triplet.split(","))))

        return ret

    def temporal_direction(self):
        if (
            self.hparams.temporal_cfg.type is None
            or not self.hparams.temporal_cfg.bidirectional
        ):
            return 1
        else:
            return 2

    def frames_feature_extractor(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
    ):
        for i in range(0, x.shape[1]):
            output[:, i, :, :] = self.feature_extractor.forward_features(
                x[:, i, :, :, :]
            )
        return output.to(self.device)

    def forward(self, x):
        output_tensor = torch.zeros(
            [x.shape[0], x.shape[1], self.vit_dim, self.feature_extractor.num_features]
        )
        feature = self.frames_feature_extractor(x, output_tensor)

        tool_seq_info = self.tool_information(feature[:, -1, :, :])

        tool_info = tool_seq_info[:, 0, :]
        tool_logit = self.tool_head(tool_info)

        attn_feature = self.attention_pre_fc(feature[:, -1, :, :])

        attn_output, _ = self.target_tool_attention(
            attn_feature,
            tool_seq_info,
            tool_seq_info,
            need_weights=True,
        )

        target_logit = self.target_head(attn_output.mean(dim=1))

        ts_feature, _ = self.ts(feature.mean(dim=2))
        ts_feature = self.ts_fc(ts_feature)
        verb_logit = self.verb_head(ts_feature[:, -1, :])
        triplet_logit = self.triplet_head(ts_feature[:, -1, :])

        return tool_logit, target_logit, verb_logit, triplet_logit

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
        tool_logit, target_logit, verb_logit, triplet_logit = self.forward(
            batch["image"]
        )
        tool_loss = self.tool_criterion(tool_logit, batch["tool"])
        target_loss = self.target_criterion(target_logit, batch["target"])
        verb_loss = self.verb_criterion(verb_logit, batch["verb"])
        triplet_loss = self.triplet_criterion(triplet_logit, batch["triplet"])
        return (
            (
                self.hparams.loss_weight.tool_weight * tool_loss
                + self.hparams.loss_weight.target_weight * target_loss
                + self.hparams.loss_weight.verb_weight * verb_loss
                + self.hparams.loss_weight.triplet_weight * triplet_loss
            )
            / (
                self.hparams.loss_weight.tool_weight
                + self.hparams.loss_weight.target_weight
                + self.hparams.loss_weight.verb_weight
                + self.hparams.loss_weight.triplet_weight
            ),
            f.softmax(tool_logit),
            f.softmax(target_logit),
            f.softmax(verb_logit),
            f.softmax(triplet_logit),
        )

    def training_step(self, batch: Any, batch_idx: int):
        loss, tool_logit, target_logit, verb_logit, triplet_logit = self.step(batch)

        # self.train_recog_metric.update(
        #     batch["triplet"].cpu().numpy(),
        #     triplet_logit,
        # )
        self.train_tool_map(tool_logit, batch["tool"].to(torch.int))
        self.train_target_map(target_logit, batch["target"].to(torch.int))
        self.train_verb_map(verb_logit, batch["verb"].to(torch.int))
        self.train_triplet_map(triplet_logit, batch["triplet"].to(torch.int))
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/tool_mAP",
            self.train_tool_map,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/verb_mAP",
            self.train_verb_map,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/target_mAP",
            self.train_target_map,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/triplet_mAP",
            self.train_triplet_map,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # self.train_tool_map.reset()
        # self.train_target_map.reset()
        # self.train_verb_map.reset()
        # self.train_triplet_map.reset()
        # ivt_result = self.train_recog_metric.compute_global_AP("ivt")
        # pprint(ivt_result["AP"])
        # self.log("train/ivt_mAP", ivt_result["mAP"])
        # self.log("train/i_mAP", self.train_recog_metric.compute_global_AP("i")["mAP"])
        # self.log("train/v_mAP", self.train_recog_metric.compute_global_AP("v")["mAP"])
        # self.log("train/t_mAP", self.train_recog_metric.compute_global_AP("t")["mAP"])
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, tool_logit, target_logit, verb_logit, triplet_logit = self.step(batch)

        self.valid_tool_map(tool_logit, batch["tool"].to(torch.int))
        self.valid_target_map(target_logit, batch["target"].to(torch.int))
        self.valid_verb_map(verb_logit, batch["verb"].to(torch.int))
        self.valid_triplet_map(triplet_logit, batch["triplet"].to(torch.int))

        self.log(
            "valid/tool_mAP",
            self.valid_tool_map,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "valid/verb_mAP",
            self.valid_verb_map,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "valid/target_mAP",
            self.valid_target_map,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "valid/triplet_mAP",
            self.valid_triplet_map,
            on_step=True,
            on_epoch=True,
        )

        tool_logit, target_logit, verb_logit, triplet_logit = (
            tool_logit.detach().cpu().numpy(),
            target_logit.detach().cpu().numpy(),
            verb_logit.detach().cpu().numpy(),
            triplet_logit.detach().cpu().numpy(),
        )

        self.valid_recog_metric.update(
            batch["triplet"].cpu().numpy(),
            triplet_logit,
        )
        self.log("valid/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        # self.valid_tool_map.reset()
        # self.valid_target_map.reset()
        # self.valid_verb_map.reset()
        # self.valid_triplet_map.reset()
        ivt_result = self.valid_recog_metric.compute_global_AP("ivt")
        pprint(ivt_result["AP"])
        self.log(
            "valid/ivt_mAP",
            ivt_result["mAP"],
        )
        self.log("valid/i_mAP", self.valid_recog_metric.compute_global_AP("i")["mAP"])
        self.log("valid/v_mAP", self.valid_recog_metric.compute_global_AP("v")["mAP"])
        self.log("valid/t_mAP", self.valid_recog_metric.compute_global_AP("t")["mAP"])

    def test_step(self, batch: Any, batch_idx: int):
        loss, tool_logit, target_logit, verb_logit, triplet_logit = self.step(batch)

        tool_logit, target_logit, verb_logit, triplet_logit = (
            tool_logit.detach().cpu().numpy(),
            target_logit.detach().cpu().numpy(),
            verb_logit.detach().cpu().numpy(),
            triplet_logit.detach().cpu().numpy(),
        )

        post_tool_logit, post_target_logit, post_verb_logit = (
            np.zeros([triplet_logit.shape[0], 100]),
            np.zeros([triplet_logit.shape[0], 100]),
            np.zeros([triplet_logit.shape[0], 100]),
        )

        for i in range(triplet_logit.shape[0]):
            for index, _triplet in enumerate(self.triplet_map):
                post_tool_logit[i][index] = tool_logit[i][_triplet[1]]
                post_verb_logit[i][index] = verb_logit[i][_triplet[2]]
                post_target_logit[i][index] = target_logit[i][_triplet[3]]

        self.test_recog_metric.update(
            batch["triplet"].cpu().numpy(),
            # triplet_logit,
            triplet_logit + 0.4 * post_target_logit + 0.2 * post_verb_logit,
        )
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def test_epoch_end(self, outputs: List[Any]):
        ivt_result = self.test_recog_metric.compute_global_AP("ivt")
        pprint(ivt_result["AP"])
        self.log(
            "test/ivt_mAP",
            ivt_result["mAP"],
        )
        self.log("test/i_mAP", self.test_recog_metric.compute_global_AP("i")["mAP"])
        self.log("test/v_mAP", self.test_recog_metric.compute_global_AP("v")["mAP"])
        self.log("test/t_mAP", self.test_recog_metric.compute_global_AP("t")["mAP"])

    def on_epoch_end(self):
        self.train_recog_metric.reset()
        self.valid_recog_metric.reset()
        self.test_recog_metric.reset()

    def configure_optimizers(self):
        opt = getattr(torch.optim, self.hparams.optim.optim_name)(
            params=self.parameters(),
            lr=self.hparams.optim.lr,
            weight_decay=self.hparams.optim.weight_decay,
        )
        lr_scheduler = getattr(
            torch.optim.lr_scheduler, self.hparams.optim.scheduler_name
        )(
            opt,
            **self.hparams.optim.scheduler,
        )
        return [opt], [lr_scheduler]
