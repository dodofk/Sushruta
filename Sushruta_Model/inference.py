import os
import torch
import json
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from torch.nn.functional import softmax
import random
from src.models.cholec_baseline_module import TripletBaselineModule
from PIL import Image
from torchvision import transforms


@hydra.main(config_path="configs/", config_name="eval.yaml")
def validation(args):
    random.seed(12345)
    data_dir = os.path.join(get_original_cwd(), "../record/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("---- Loading Model ----")
    model = TripletBaselineModule.load_from_checkpoint(
        os.path.join(get_original_cwd(), args.ckpt_path)
    ).to(device)

    model.eval()
    print("---- Finish Loading ----")

    videos = os.listdir(data_dir)

    transform_fn = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    for video in videos:
        video_record = list()
        with open(os.path.join("../data/", f"{video}.txt"), "r") as f:
            data = f.read().split("\n")
        if "" in data:
            data.remove("")

        prev_img = None
        with torch.no_grad():
            for _data in data:
                image = Image.open(
                    os.path.join(get_original_cwd(), "..", _data),
                )
                image = transform_fn(image)
                if prev_img is None:
                    prev_img = image

                frames = torch.FloatTensor(1, 2, 3, 224, 224)

                frames[:, 0, :, :, :] = prev_img
                frames[:, 1, :, :, :] = image

                tool_logit, target_logit, verb_logit, triplet_logit = model(
                    frames.to(device)
                )
                tool_logit, target_logit, verb_logit, triplet_logit = (
                    softmax(tool_logit, dim=-1).detach().cpu().numpy(),
                    softmax(target_logit, dim=-1).detach().cpu().numpy(),
                    softmax(verb_logit, dim=-1).detach().cpu().numpy(),
                    softmax(triplet_logit, dim=-1).detach().cpu().numpy(),
                )

                post_tool_logit, post_target_logit, post_verb_logit = (
                    np.zeros([triplet_logit.shape[0], 100]),
                    np.zeros([triplet_logit.shape[0], 100]),
                    np.zeros([triplet_logit.shape[0], 100]),
                )

                for i in range(triplet_logit.shape[0]):
                    for index, _triplet in enumerate(model.triplet_map):
                        post_tool_logit[i][index] = tool_logit[i][_triplet[1]]
                        post_verb_logit[i][index] = verb_logit[i][_triplet[2]]
                        post_target_logit[i][index] = target_logit[i][_triplet[3]]

                combined_triplet_logit = (
                    0.5 * post_tool_logit
                    + 1.25 * post_verb_logit * post_target_logit
                    + 1 * triplet_logit
                )
                frame_id: str = _data.split("/")[-1]
                video_record[frame_id] = {
                    "recognition": combined_triplet_logit,
                    "detection": None
                }
        with open(os.path.join(get_original_cwd(), "../output", f"{video}.json"), "w") as f:
            json.dump(video_record, f)


if __name__ == "__main__":
    validation()
