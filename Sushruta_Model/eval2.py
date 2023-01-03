import os
import torch
import json
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from statistics import stdev, mean
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import ivtmetrics
import random

from torchmetrics import Precision
from pprint import pprint
from src.datamodules.components.cholect45_dataset import (
    CholecT45Dataset,
    default_collate_fn,
)
from src.models.cholec_baseline_module import TripletBaselineModule


VALIDATION_VIDEOS = ["78", "43", "62", "35", "74", "01", "56", "04", "13"]
MINORITY_CLASSES = ["97", "91", "01","48", "53", "50", "26", "46", "33", "90", "05", "89", "04", "51","72","44","98", "09", "34", "18", "52", "35", "83", "99", "92", "24", "75", "21", "22", "31"]  # top 30

@hydra.main(config_path="configs/", config_name="eval2.yaml")
def validation(args):
    random.seed(args.seed)
    data_dir = os.path.join(get_original_cwd(), "data/CholecT45/")
    device = args.device if torch.cuda.is_available() else "cpu"

    with open(os.path.join(get_original_cwd(), "data/triplet_class_arg.npy"), "rb") as file:
        triplet_sort_ind = np.load(file)

    valid_record = dict()

    print("---- Loading Model ----")
    model = TripletBaselineModule.load_from_checkpoint(
        os.path.join(get_original_cwd(), args.ckpt_path)
    ).to(device)
    model.eval()
    print("---- Finish Loading ----")
    global_ivt_metric = ivtmetrics.Recognition(num_class=100)
    global_ivt_metric.reset_global()

    for video in VALIDATION_VIDEOS:
        print(f"Video: {video}")
        dataset = CholecT45Dataset(
            img_dir=os.path.join(data_dir, "data", f"VID{video}"),
            triplet_file=os.path.join(data_dir, "triplet", "VID{}.txt".format(video)),
            tool_file=os.path.join(data_dir, "instrument", "VID{}.txt".format(video)),
            verb_file=os.path.join(data_dir, "verb", "VID{}.txt".format(video)),
            target_file=os.path.join(data_dir, "target", "VID{}.txt".format(video)),
            split="dev",
            data_dir=data_dir,
            seq_len=2,
            channels=3,
            use_train_aug=False,
            triplet_class_arg=os.path.join(
                get_original_cwd(), "data/triplet_class_arg.npy"
            ),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=default_collate_fn,
            num_workers=0,
            shuffle=False,
        )
        triplet_map = Precision(
            num_classes=model.class_num["triplet"],
            average="macro",
        )
        tool_map = Precision(
            num_classes=model.class_num["tool"],
            average="macro",
        )
        verb_map = Precision(
            num_classes=model.class_num["verb"],
            average="macro",
        )
        target_map = Precision(
            num_classes=model.class_num["target"],
            average="macro",
        )
        ivt_metric = ivtmetrics.Recognition(num_class=100)

        with torch.no_grad():
            for batch in tqdm(dataloader):  # batch.keys() ->['frame', 'image', 'verb', 'tool', 'target', 'triplet']
                tool_logit, target_logit, verb_logit, triplet_logit = model(
                    batch["image"].to(args.device)
                )

                triplet_map(triplet_logit.to("cpu"), batch["triplet"].to(torch.int))
                tool_map(tool_logit.to("cpu"), batch["tool"].to(torch.int))
                verb_map(verb_logit.to("cpu"), batch["verb"].to(torch.int))
                target_map(target_logit.to("cpu"), batch["target"].to(torch.int))



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

                combined_triplet_logit = 0.5 * post_tool_logit + 1.25 * post_verb_logit * post_target_logit + 1 * triplet_logit

                arg_max = np.argmax(combined_triplet_logit, axis=1)

                print(f'arg_max:\n{arg_max}')

                if args.random_aug:
                    for index, _arg_max in enumerate(arg_max):
                        if _arg_max in triplet_sort_ind[-7:]:
                            if random.random() < args.rand_ratio:
                                for max_idx in triplet_sort_ind[-7:]:
                                    combined_triplet_logit[index][max_idx] = 0
                                randn_idx = random.choice(triplet_sort_ind[5:40])
                                combined_triplet_logit[index][randn_idx] = 15

                ivt_metric.update(
                    batch["triplet"].cpu().numpy(),
                    combined_triplet_logit,
                )
                global_ivt_metric.update(
                    batch["triplet"].cpu().numpy(),
                    combined_triplet_logit,
                )
        
        valid_record[video] = {
            "triplet": triplet_map.compute().item(),
            "tool": tool_map.compute().item(),
            "verb": verb_map.compute().item(),
            "target": target_map.compute().item(),
            "i_mAP": ivt_metric.compute_global_AP("i")["mAP"],
            "v_mAP": ivt_metric.compute_global_AP("v")["mAP"],
            "t_mAP": ivt_metric.compute_global_AP("t")["mAP"],
            "ivt_mAP": ivt_metric.compute_global_AP("ivt")["mAP"],
            "ivt_detail": list(ivt_metric.compute_global_AP("ivt")["AP"]),
        }
        global_ivt_metric.video_end()

    valid_record["overall"] = {
        "triplet": {
            "mean": mean([record["triplet"] for record in valid_record.values()]),
            "stdev": stdev([record["triplet"] for record in valid_record.values()]),
        },
        "tool": {
            "mean": mean([record["tool"] for record in valid_record.values()]),
            "stdev": stdev([record["tool"] for record in valid_record.values()]),
        },
        "verb": {
            "mean": mean([record["verb"] for record in valid_record.values()]),
            "stdev": stdev([record["verb"] for record in valid_record.values()]),
        },
        "target": {
            "mean": mean([record["target"] for record in valid_record.values()]),
            "stdev": stdev([record["target"] for record in valid_record.values()]),
        },
        "i_mAP": {
            "mean": mean([record["i_mAP"] for record in valid_record.values()]),
            "stdev": stdev([record["i_mAP"] for record in valid_record.values()]),
        },
        "v_mAP": {
            "mean": mean([record["v_mAP"] for record in valid_record.values()]),
            "stdev": stdev([record["v_mAP"] for record in valid_record.values()]),
        },
        "t_mAP": {
            "mean": mean([record["t_mAP"] for record in valid_record.values()]),
            "stdev": stdev([record["t_mAP"] for record in valid_record.values()]),
        },
        "ivt_mAP": {
            "mean": mean([record["ivt_mAP"] for record in valid_record.values()]),
            "stdev": stdev([record["ivt_mAP"] for record in valid_record.values()]),
        },
        "g_i_mAP": global_ivt_metric.compute_video_AP("i")["mAP"],
        "g_v_mAP": global_ivt_metric.compute_video_AP("v")["mAP"],
        "g_t_mAP": global_ivt_metric.compute_video_AP("t")["mAP"],
        "g_ivt_mAP": global_ivt_metric.compute_video_AP("ivt")["mAP"],
        "g_ivt_detail": list(global_ivt_metric.compute_video_AP("ivt")["AP"]),
    }

    with open(os.path.join(get_original_cwd(), args.output_fname), "w") as f:
        json.dump(valid_record, f, sort_keys=True, indent=4)

    pprint(valid_record)

@hydra.main(config_path="configs/", config_name="eval2.yaml")
def testtest(args):

    VALIDATION_VIDEOS = ["78", "43", "62", "35", "74", "01", "56", "04", "13"]

    i_map = {
        '0': "grasper",
        '1': "bipolar",
        '2': "hook",
        '3': "scissors",
        '4': "clipper",
        '5': "irrigator",
        '-1': "null_instrument",
    }

    v_map = {
        '0': "grasp",
        '1': "retract",
        '2': "dissect",
        '3': "coagulate",
        '4': "clip",
        '5': "cut",
        '6': "aspirate",
        '7': "irrigate",
        '8': "pack",
        '9': "null_verb",
    }

    t_map = {
        '0': "gallbladder",
        '1': "cystic_plate",
        '2': "cystic_duct",
        '3': "cystic_artery",
        '4': "cystic_pedicle",
        '5': "blood_vessel",
        '6': "fluid",
        '7': "abdominal_wall_cavity",
        '8': "liver",
        '9': "adhesion",
        '10': "omentum",
        '11': "peritoneum",
        '12': "gut",
        '13': "specimen_bag",
        '14': "null_target",
    }

    device = args.device if torch.cuda.is_available() else "cpu"
    
    print("---- Loading Model ----") 
    model = TripletBaselineModule.load_from_checkpoint(
        os.path.join(get_original_cwd(), args.ckpt_path)
    ).to(device)
    model.eval()
    print("---- Finish Loading ----")
    global_ivt_metric = ivtmetrics.Recognition(num_class=100)
    global_ivt_metric.reset_global()
    
    wrong_logit_i, wrong_logit_v, wrong_logit_t = {}, {}, {}
    wrong_pred_i, wrong_pred_v, wrong_pred_t = {}, {}, {}

    # for testing purposes only
    # video = VALIDATION_VIDEOS[0]
    for video in tqdm(VALIDATION_VIDEOS):
        print(f'Processing video {video}...'.ljust(50), end='\r')
        data_dir = os.path.join(get_original_cwd(), "data/CholecT45/")
        dataset = CholecT45Dataset(
                img_dir=os.path.join(data_dir, "data", f"VID{video}"),
                triplet_file=os.path.join(data_dir, "triplet", "VID{}.txt".format(video)),
                tool_file=os.path.join(data_dir, "instrument", "VID{}.txt".format(video)),
                verb_file=os.path.join(data_dir, "verb", "VID{}.txt".format(video)),
                target_file=os.path.join(data_dir, "target", "VID{}.txt".format(video)),
                split="dev",
                data_dir=data_dir,
                seq_len=2,
                channels=3,
                use_train_aug=False,
                triplet_class_arg=os.path.join(
                    get_original_cwd(), "data/triplet_class_arg.npy"
                ),
            )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=default_collate_fn,
            num_workers=0,
            shuffle=False,
        )

        for batch in tqdm(dataloader):
            tool_logit, target_logit, verb_logit, triplet_logit = model(
                batch["image"].to(args.device)
            )


            tool_logit, target_logit, verb_logit, triplet_logit = (
                softmax(tool_logit, dim=-1).detach().cpu().numpy(),
                softmax(target_logit, dim=-1).detach().cpu().numpy(),
                softmax(verb_logit, dim=-1).detach().cpu().numpy(),
                softmax(triplet_logit, dim=-1).detach().cpu().numpy(),
            )
            pred_tool_index, pred_verb_index, pred_target_index, pred_triplet_index = (
                np.argmax(tool_logit, axis=1),
                np.argmax(verb_logit, axis=1),
                np.argmax(target_logit, axis=1),
                np.argmax(triplet_logit, axis=1)
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

            combined_triplet_logit = 0.5 * post_tool_logit + 1.25 * post_verb_logit * post_target_logit + 1 * triplet_logit
            
            # modified: 
            # mix_pred_triplet_index = [str(j) for j in range(len(mix_pred_triplet) if mix_pred_triplet(j) > threshold)]
            mix_pred_triplet_index = [str(idx) for idx in np.argmax(combined_triplet_logit, axis=1)]

            for i, arr in enumerate(batch['triplet']):
                truth_triplet_indexs = [str(j) for j in range(len(arr)) if arr[j] == 1]
                truth_i_lst, truth_v_lst, truth_t_lst = [], [], []

                # finding truth triplet corresponding i, v, t (only miniority triplet)
                for trip_id in truth_triplet_indexs:
                    if trip_id in MINORITY_CLASSES:
                        truth_i_lst.append(str(model.triplet_map[int(trip_id)][1]))
                        truth_v_lst.append(str(model.triplet_map[int(trip_id)][2]))
                        truth_t_lst.append(str(model.triplet_map[int(trip_id)][3]))

                # logit i, v, t, predictions analysis
                if not str(pred_tool_index[i]) in truth_i_lst:
                    for truth_i in truth_i_lst:
                        wrong_i_key = f'{i_map[truth_i]} --> {i_map[str(pred_tool_index[i])]}'
                        if not wrong_i_key in wrong_logit_i.keys():
                            wrong_logit_i[wrong_i_key] = 0
                        wrong_logit_i[wrong_i_key] += 1
                if not str(pred_verb_index[i]) in truth_v_lst:
                    for truth_v in truth_v_lst:
                        wrong_v_key = f'{v_map[truth_v]} --> {v_map[str(pred_verb_index[i])]}'
                        if not wrong_v_key in wrong_logit_v.keys():
                            wrong_logit_v[wrong_v_key] = 0
                        wrong_logit_v[wrong_v_key] += 1
                if not str(pred_target_index[i]) in truth_t_lst:
                    for truth_t in truth_t_lst:
                        wrong_t_key = f'{t_map[truth_t]} --> {t_map[str(pred_target_index[i])]}'
                        if not wrong_t_key in wrong_logit_t.keys():
                            wrong_logit_t[wrong_t_key] = 0
                        wrong_logit_t[wrong_t_key] += 1
                
                
                # mix predictions analysis
                mix_pred_i, mix_pred_v, mix_pred_t = (
                    str(model.triplet_map[int(mix_pred_triplet_index[i])][1]),
                    str(model.triplet_map[int(mix_pred_triplet_index[i])][2]),
                    str(model.triplet_map[int(mix_pred_triplet_index[i])][3]),
                )
                if not mix_pred_i in truth_i_lst:
                    for truth_i in truth_i_lst:
                        wrong_i_key = f'{i_map[truth_i]} --> {i_map[mix_pred_i]}'
                        if not wrong_i_key in wrong_pred_i.keys():
                            wrong_pred_i[wrong_i_key] = 0
                        wrong_pred_i[wrong_i_key] += 1
                if not mix_pred_v in truth_v_lst:
                    for truth_v in truth_v_lst:
                        wrong_v_key = f'{v_map[truth_v]} --> {v_map[mix_pred_v]}'
                        if not wrong_v_key in wrong_pred_v.keys():
                            wrong_pred_v[wrong_v_key] = 0
                        wrong_pred_v[wrong_v_key] += 1
                if not mix_pred_t in truth_t_lst:
                    for truth_t in truth_t_lst:
                        wrong_t_key = f'{t_map[truth_t]} --> {t_map[mix_pred_t]}'
                        if not wrong_t_key in wrong_pred_t.keys():
                            wrong_pred_t[wrong_t_key] = 0
                        wrong_pred_t[wrong_t_key] += 1
    print('-------- Logit --------')
    print('--- Wrong Predicted Instrument ---')
    for k, v in sorted(wrong_logit_i.items(), key=lambda x: x[1], reverse=True):
        print(f'{k}: {str(v).zfill(3)}')
    print()
    print('--- Wrong Predicted Verb ---')
    for k, v in sorted(wrong_logit_v.items(), key=lambda x: x[1], reverse=True):
        print(f'{k}: {str(v).zfill(3)}')
    print()
    print('--- Wrong Predicted Target ---')
    for k, v in sorted(wrong_logit_t.items(), key=lambda x: x[1], reverse=True):
        print(f'{k}: {str(v).zfill(3)}')
    print('\n'*2)
    print('-------- Mix --------')          
    print('--- Wrong Predicted Instrument ---')
    for k, v in sorted(wrong_pred_i.items(), key=lambda x: x[1], reverse=True):
        print(f'{k}: {str(v).zfill(3)}')
    print()
    print('--- Wrong Predicted Verb ---')
    for k, v in sorted(wrong_pred_v.items(), key=lambda x: x[1], reverse=True):
        print(f'{k}: {str(v).zfill(3)}')
    print()
    print('--- Wrong Predicted Target ---')
    for k, v in sorted(wrong_pred_t.items(), key=lambda x: x[1], reverse=True):
        print(f'{k}: {str(v).zfill(3)}')
    print()


        




if __name__ == "__main__":
    # validation()
    testtest()



# python eval2.py ckpt_path=logs/experiments/runs/cholect45_basic_andy/2022-08-02_14-20-24/checkpoints/epoch_009.ckpt output_fname=andytest1.json