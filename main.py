import os
import sys
import time
import random
import progressbar

from convert_npy import *
from run_model import *
from glob import glob

_BAR = progressbar.ProgressBar()
_ENABLE_PRINT = True
_VEDIO_PATH = "vedios"
_OUT_FILE = "time.txt"
_INIT_FILE = "vedios/basketball.avi"
_VEDIO_COUNT = 1000
_DIVID_COUNT = 50

# 取消所有print
if not _ENABLE_PRINT:
    f = open(os.devnull, 'w')
    sys.stdout = f

# 只输出error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# init rgb_only_64_frames i3d model
model = None


def del_path(filepath):
    """ 删除某一目录下的所有文件夹 """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)


def init():
    global model
    model = I3D_RGB()
    init_file = _INIT_FILE
    success, rgb_npy_file = vedio_to_rgb_npy(init_file, save_file=False)
    model.run(rgb_npy_file)


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
    all_time, count, index, length = 0.0, 0, 0, len(vedios)
    print("Start run batch test")
    _BAR.start()
    while count < _VEDIO_COUNT and index < length:
        _BAR.update(count * 100 / _VEDIO_COUNT)
        vedio = vedios[index]
        activity = os.path.basename(vedio).split(".")[0]
        success, rgb_npy_file = vedio_to_rgb_npy(vedio, save_file=False)
        if not success:
            index += 1
            continue
        start = time.clock()
        model.run(rgb_npy_file)
        end = time.clock()
        all_time += end - start
        count += 1
        index += 1
        if count % _DIVID_COUNT == 0 and count > 0 and success:
            result.append("batch size : {:<10d} time : {:.05f}s\n".format(
                count, all_time))
    _BAR.finish()
    print("See result in {}".format(_OUT_FILE))
    with open(_OUT_FILE, "w") as f:
        f.writelines(result)