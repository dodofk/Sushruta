#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CODE RELEASE FOR CHOLECTRIPLET2022 DEMO PURPOSE ONLY.
This code is not shareable.
This code is not released to the public domain.
#==============================================================================
An implementation from:
***
    C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. 
    Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. 
    Medical Image Analysis 78 (2022) 102433.
***  
Created on Wed Apr 20 15:50:52 2022
#==============================================================================  
Copyright 2019 The Research Group CAMMA Authors All Rights Reserved.
(c) Research Group CAMMA, University of Strasbourg, France
@ Laboratory: CAMMA - ICube
@ Author: Nwoye Chinedu
@ Website: http://camma.u-strasbg.fr
#==============================================================================
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#==============================================================================
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont


class CholecT50Dataset(object):
    def __init__(self, labelpath, videopath, mapfile="./id2names.txt"):
        # self.labelpath = labelpath
        self.videopath = videopath
        self.loadJSON(labelpath)
        self.loadNames(mapfile)

    def loadNames(self, mapfile):
        self.names = {}
        with open(mapfile) as stream:
            for txt in stream:
                k, v = txt.split(":")
                v = v.replace(",", ", ").replace("\n", "")
                self.names[int(k)] = v
        stream.close()

    def loadJSON(self, labelpath):
        self.data = json.load(open(labelpath, "rb"))

    def get_labels(self, frameid):
        return self.data[str(frameid)]

    def get_image(self, frameid):
        imgurl = os.path.join(self.videopath, "{}.png".format(str(frameid).zfill(6)))
        img = Image.open(imgurl)
        return img

    def show_image_and_labels(self, frameid):
        labels = self.get_labels(frameid)
        img = self.get_image(frameid)
        w, h = img.size
        print(h, w)
        # h,w = 434, 774
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("./Verdana.ttf", 18)
        for label in labels:
            xy = label["instrument"][-4:]
            text = self.names[label["triplet"]]
            x0 = xy[0] * w
            y0 = xy[1] * h
            x1 = (xy[0] + xy[2]) * w
            y1 = (xy[1] + xy[3]) * h
            draw.rectangle([(x0, y0), (x1, y1)], fill=None, outline="blue", width=1)
            st = x0 + 10 if x1 < 600 else x0 - 30
            draw.text(
                (st, (y0 + y1) // 2), text, fill="yellow", align="center", font=font
            )
        img.show()

    def list_frames(self):
        return list(map(int, self.data.keys()))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--list-frames",
        action="store_true",
        help="To display the IDs of all image frames in the video directory.",
    )
    parser.add_argument(
        "--frameid",
        type=int,
        default=-1,
        help="provide the frame iD (int) to visualize.",
    )
    parser.add_argument(
        "-sl",
        "--show-label",
        action="store_true",
        help="To display the labels of image of the provided frame ID.",
    )
    parser.add_argument(
        "-si",
        "--show-image",
        action="store_true",
        help="To display the image of the provided frame ID.",
    )
    parser.add_argument(
        "-sil",
        "--show-image-and-label",
        action="store_true",
        help="To display an image with the corresponding labels of the provided frame ID.",
    )
    FLAGS, unparsed = parser.parse_known_args()

    labelpath = "./label.json"
    videopath = "./video/"
    mapfile = "./id2names.txt"
    obj = CholecT50Dataset(labelpath, videopath, mapfile)
    frames = obj.list_frames()
    frameid = FLAGS.frameid
    frameid = frames[0] if frameid == -1 else frameid

    #%%%%%%%%%%%% To see the list of frame ids, run: %%%%%%%%%%%%%%%%#%%%%%%%#%%%%%%%%%
    if FLAGS.list_frames:
        print("FRAME IDS:\n\t", frames, "\n", "%" * 100)

    #%%%%%%%%%%%% To display the labels in one frame, run: %%%%%%%%%%%%%%%%#%%%%%%%%%%%
    if FLAGS.show_label:
        print(
            "LABEL FORMAT:\n",
            ' [\n\t{"triplet":tripletID, "instrument":[toolID, toolProbs, x, y, w, h]}, \n\t{"triplet":tripletID, "instrument":[toolID, toolProbs, x, y, w, h]}, \n\t...\n]\n',
            "%" * 40,
            "RESULTS",
            "%" * 40,
        )
        labels = obj.get_labels(frameid)
        print(labels)

    #%%%%%%%%%%%% To display the image data of a chosen frame id, run: %%%%%%%%%%%%%%%%
    if FLAGS.show_image:
        img = obj.get_image(frameid)
        img.show()

    #%%%%%%%%%%%% To display an image with the corresponding labels, run: %%%%%%%%%%%%%
    if FLAGS.show_image_and_label:
        obj.show_image_and_labels(frameid)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print(
        "\n\nRun with -h option for help...\n",
    )
