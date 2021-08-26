import os
import sys
import time
import random

from convert_npy import *
from run_model import *
from glob import glob

_ENABLE_PRINT = False
_VEDIO_PATH = "vedios"
_OUT_FILE = "time.txt"
_INIT_FILE = "vedios/basketball.avi"
_VEDIO_COUNT = 200

# 取消所有print
if not _ENABLE_PRINT:
    f = open(os.devnull, 'w')
    sys.stdout = f

# 只输出error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# init rgb_only_64_frames i3d model
model = None


def del_path(filepath):
    """ 删除某一目录下的所有文件夹 """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)


def batch_test(vedio_count, vedio_path=_VEDIO_PATH):
    # get vedio file path
    vedios = glob(os.path.join(os.getcwd(), vedio_path, "*.avi"))
    if len(vedios) < vedio_count:
        print("vedios count not enough : ", len(vedios))
        return False
    random.shuffle(vedios)
    for file_path in vedios:
        if vedio_count == 0:
            break
        activity = os.path.basename(file_path).split(".")[0]
        # convert vedio file to rgb_64_frame npy file
        rgb_npy_file = vedio_to_rgb_npy(file_path)
        if rgb_npy_file != None:
            out_path = os.path.join(os.getcwd(), "vedios", activity)
            # get prediction result > out_path/out.txt
            model.run(rgb_npy_file, out_path)
        vedio_count -= 1
    return True


def init():
    global model
    model = I3D_RGB()
    init_file = _INIT_FILE
    init_npy_file = generate_rgb64_data(init_file)
    model.run(init_npy_file)


if __name__ == "__main__":
    result = []
    start = time.clock()
    init()
    end = time.clock()
    result.append("I3D Model Init : {:.05f}s\n".format(end - start))
    vedios = glob(os.path.join(os.getcwd(), _VEDIO_PATH, "*.avi"))
    if len(vedios) < _VEDIO_COUNT:
        print("vedios count not enough : ", len(vedios))
        exit()
    random.shuffle(vedios)
    count = 0
    start = time.clock()
    while count<_VEDIO_COUNT:
        vedio = vedios[count]
        activity = os.path.basename(vedio).split(".")[0]
        rgb_npy_file = vedio_to_rgb_npy(vedio)
        if rgb_npy_file != None:
            out_path = os.path.join(os.getcwd(), "vedios", activity)
            # get prediction result > out_path/out.txt
            model.run(rgb_npy_file, out_path)
        count+=1
        if count==1 or count%5==0:
            end = time.clock()
            result.append("batch size : {:<10d} time : {:.05f}s\n".format(
                count, end - start))
    with open(_OUT_FILE, "w") as f:
        f.writelines(result)