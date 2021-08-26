import os
import random

# 只输出error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from convert_npy import *
from run_model import *
from glob import glob


def batch_test(vedio_count, vedio_path="vedios"):
    # get vedio file path
    vedios = glob(os.path.join(os.getcwd(), vedio_path, "*.avi"))
    if len(vedios) < vedio_count:
        print("vedios count not enough : ", len(vedios))
        return
    random.shuffle(vedios)
    # init rgb_only_64_frames i3d model
    model = I3D_RGB()
    for file_path in vedios:
        activity = os.path.basename(file_path).split(".")[0]
        # convert vedio file to rgb_64_frame npy file
        rgb_npy_file = vedio_to_rgb_npy(file_path)
        if rgb_npy_file != "":
            out_path = os.path.join(os.getcwd(), "vedios", activity)
            # get prediction result > out_path/out.txt
            model.run(rgb_npy_file, out_path)


batch_test(vedio_count=5)