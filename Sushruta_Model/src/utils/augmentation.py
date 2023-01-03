import PIL.Image as Image
import warnings
import random
import time
import os

warnings.filterwarnings("ignore")
from torchvision import transforms

PHASE_CNT = 7
TRAIN_LABEL_PATH = "train.csv"
VIDEO_NAME = "HeiChole"
AUG_PHASE = ("4", "5", "6")
# if you want to put new image in original file, choose False
CREATE_AUG_DIR = True
# if you want to put new images' label in original csv, choose False
CREATE_AUG_LABEL = True
# if you wnat the new label csv include original images' label, choose True
INCLUDE_ORG_LABEL = True

picture_size = 256

random.seed(time.time())
padding = (
    int(random.uniform(0, 0.05) * picture_size),
    int(random.uniform(0, 0.05) * picture_size),
    int(random.uniform(0, 0.05) * picture_size),
    int(random.uniform(0, 0.05) * picture_size),
)

transform_set = [
    # transforms.Pad(padding, padding_mode="edge"),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomVerticalFlip(p=0.6),
    transforms.RandomRotation(
        degrees=(0, 360),
        expand=False,
        center=(256 / 2 + random.randint(-10, 10), 256 / 2 + random.randint(-10, 10)),
    ),
    transforms.RandomPerspective(distortion_scale=0.6, p=0.8, interpolation=2),
    transforms.ColorJitter(
        brightness=(1, 3), contrast=(1, 10), saturation=(1, 5), hue=(-0.1, 0.1)
    ),
]

transform = transforms.Compose(
    [
        # transforms.RandomChoice(transform_set),
        # transforms.RandomChoice(transform_set)
        transforms.RandomOrder(transform_set)
    ]
)

aug_pic_dic = {}
is_header = True
re_write_infor = []

with open(TRAIN_LABEL_PATH, "r") as f:
    lines = f.readlines()
    for line in lines:
        if is_header:
            is_header = False
            continue
        else:
            video_no, img_id, phase_no = line.split(",")
            phase_no = phase_no.replace("\n", "")

            if INCLUDE_ORG_LABEL:
                re_write_infor.append([video_no, img_id, phase_no])

            if phase_no in AUG_PHASE:
                aug_pic_dic[f"{video_no}/{img_id}"] = phase_no

# aug_pic_dic = {'12/1203': '5'}
# only include trainning dataset's video


counter = 1
for img_infor in aug_pic_dic:

    # convert img into PIL image
    video_no, img_id = img_infor.split("/")
    img_pil = Image.open(f"{VIDEO_NAME}_{video_no}/{img_id}.jpg", mode="r")
    img_pil = img_pil.convert("RGB")

    new_img = transform(img_pil)

    # save new image (in a new directory ?)
    if CREATE_AUG_DIR:
        directory = f"{VIDEO_NAME}_{video_no}_A"
        try:
            os.makedirs(directory)
        except FileExistsError:
            Image.Image.save(new_img, fp=f"{VIDEO_NAME}_{video_no}_A/{img_id}_aug.jpg")
        Image.Image.save(new_img, fp=f"{VIDEO_NAME}_{video_no}_A/{img_id}_aug.jpg")
    else:
        Image.Image.save(new_img, fp=f"{VIDEO_NAME}_{video_no}/{img_id}_aug.jpg")

    # sav new image [video_no, img_id, phase_no] information:
    img_phase = aug_pic_dic[img_infor]
    aug_img_infor = [video_no, img_id + "_aug", img_phase]

    re_write_infor.append(aug_img_infor)
    print(f"Already Augmented: {counter}", end="\r")
    counter += 1

# if create new train_aug.csv, it inludes origin image's information
if CREATE_AUG_LABEL:
    with open(TRAIN_LABEL_PATH.replace(".csv", "_aug.csv"), "w") as f:
        for infor in re_write_infor:
            f.write(f"{infor[0]},{infor[1]},{infor[2]}\n")
else:
    with open(TRAIN_LABEL_PATH, "a") as f:
        for infor in re_write_infor:
            f.write(f"{infor[0]},{infor[1]},{infor[2]}\n")
