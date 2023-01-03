#!/bin/bash

gdown https://drive.google.com/uc?id=14o5HHzK7kSXQxeoOijiPpp_nHl4yer_t

mv CholecT45_resize.tar.gz data

cd data || return

tar -xvf CholecT45_resize.tar.gz

mv CholecT45_resize CholecT45
#
#unzip CholecT45_resize.zip -O
#mv CholecT45_resize.zip data/Cholec
#
#cd data/HeiChole_data && unzip HeiChole.zip