from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import i3d
import numpy as np
import tensorflow as tf

_IMAGE_SIZE = 224
_FRAME_COUNT = 64
_NUM_CLASSES = 400

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'


class I3D_RGB():
    ''' 只采用RGB通道，固定64Frame '''
    def __init__(self):
        print("I3D model init...")
        self.kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
        self.rgb_input = tf.placeholder(tf.float32,
                                        shape=(1, _FRAME_COUNT, _IMAGE_SIZE,
                                               _IMAGE_SIZE, 3))
        with tf.variable_scope('RGB'):
            self.rgb_model = i3d.InceptionI3d(_NUM_CLASSES,
                                              spatial_squeeze=True,
                                              final_endpoint='Logits')
            self.rgb_logits, _ = self.rgb_model(self.rgb_input,
                                                is_training=False,
                                                dropout_keep_prob=1.0)
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        self.rgb_saver = tf.train.Saver(var_list=rgb_variable_map,
                                        reshape=True)
        self.model_predictions = tf.nn.softmax(self.rgb_logits)

        with tf.Session().as_default() as self.sess:
            self.rgb_saver.restore(self.sess,
                                   _CHECKPOINT_PATHS['rgb_imagenet'])
        print("init finished\n")

    def run(self, rgb_npy_file, out_path=None):
        if not os.path.exists(rgb_npy_file):
            print("npy file not exists : ", rgb_npy_file)
            return
        print("predicting file : ", rgb_npy_file)
        feed_dict = {}
        rgb_sample = np.load(rgb_npy_file)
        feed_dict[self.rgb_input] = rgb_sample
        out_logits, out_predictions = self.sess.run(
            [self.rgb_logits, self.model_predictions], feed_dict=feed_dict)
        out_logits = out_logits[0]
        out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions)[::-1]

        if out_path != None:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            with open(os.path.join(out_path, "out.txt"), "w") as f:
                f.write(
                    'Norm of logits: %f\n\nTop classes and probabilities\n' %
                    np.linalg.norm(out_logits))
                f.writelines([
                    "{:.6e}\t{:.6f}\t{}\n".format(out_predictions[index],
                                                  out_logits[index],
                                                  self.kinetics_classes[index])
                    for index in sorted_indices
                ])
        else:
            print('Norm of logits: %f\nTop classes and probabilities' %
                  np.linalg.norm(out_logits))
            for index in sorted_indices[:20]:
                print(out_predictions[index], out_logits[index],
                      self.kinetics_classes[index])
        print("end\n")


def judge_vedio(frame_count,
                rgb_file,
                flow_file,
                imagenet_pretrained=True,
                eval_type="joint",
                out_path=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    NUM_CLASSES = 400
    if eval_type == 'rgb600':
        NUM_CLASSES = 600

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError(
            'Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(tf.float32,
                                   shape=(1, frame_count, _IMAGE_SIZE,
                                          _IMAGE_SIZE, 3))

        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(NUM_CLASSES,
                                         spatial_squeeze=True,
                                         final_endpoint='Logits')
            rgb_logits, _ = rgb_model(rgb_input,
                                      is_training=False,
                                      dropout_keep_prob=1.0)

        rgb_variable_map = {}
        for variable in tf.global_variables():

            if variable.name.split('/')[0] == 'RGB':
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(
                        ':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    rgb_variable_map[variable.name.replace(':0',
                                                           '')] = variable

            rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(tf.float32,
                                    shape=(1, frame_count, _IMAGE_SIZE,
                                           _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(NUM_CLASSES,
                                          spatial_squeeze=True,
                                          final_endpoint='Logits')
            flow_logits, _ = flow_model(flow_input,
                                        is_training=False,
                                        dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    if eval_type == 'rgb' or eval_type == 'rgb600':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)

    with tf.Session() as sess:
        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')
            rgb_sample = np.load(rgb_file)
            tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
            feed_dict[rgb_input] = rgb_sample

        if eval_type in ['flow', 'joint']:
            if imagenet_pretrained:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            else:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
            tf.logging.info('Flow checkpoint restored')
            flow_sample = np.load(flow_file)
            tf.logging.info('Flow data loaded, shape=%s',
                            str(flow_sample.shape))
            feed_dict[flow_input] = flow_sample

        out_logits, out_predictions = sess.run(
            [model_logits, model_predictions], feed_dict=feed_dict)

        out_logits = out_logits[0]
        out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions)[::-1]

        print('Norm of logits: %f' % np.linalg.norm(out_logits))
        print('\nTop classes and probabilities')
        for index in sorted_indices[:20]:
            print(out_predictions[index], out_logits[index],
                  kinetics_classes[index])
        if out_path != None:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            with open(os.path.join(out_path, "out.txt"), "w") as f:
                f.write(
                    'Norm of logits: %f\n\nTop classes and probabilities\n' %
                    np.linalg.norm(out_logits))
                f.writelines([
                    "{:.6e}\t{:.6f}\t{}\n".format(out_predictions[index],
                                                  out_logits[index],
                                                  kinetics_classes[index])
                    for index in sorted_indices
                ])
