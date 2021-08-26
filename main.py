import os
import random

# 只输出error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from convert_npy import *
from run_model import *
from glob import glob


def batch_test(vedio_count, vedio_path="vedios"):
    vedios = glob(os.path.join(os.getcwd(), vedio_path, "*.avi"))
    if len(vedios) < vedio_count:
        print("vedios count not enough : ", len(vedios))
        return
    random.shuffle(vedios)
    model = I3D_RGB()
    for file_path in vedios:
        activity = os.path.basename(file_path).split(".")[0]
        rgb_npy_file = vedio_to_rgb_npy(file_path)
        if rgb_npy_file != "":
            out_path = os.path.join(os.getcwd(), "vedios", activity)
            model.run(rgb_npy_file, out_path)


batch_test(vedio_count=5)