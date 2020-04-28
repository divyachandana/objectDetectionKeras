#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
import progressbar
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
import glob
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.utils import multi_gpu_model
gpu_num=1
isDrawBox = True

class YOLO(object):
    def __init__(self):
        # Huan
        # self.model_path = r'model_data/RS/yolo.h5.h5'
        # self.anchors_path = r'model_data\house_anchors.txt'
        # self.classes_path = r'D:\YOLO\keras-yolo3-master\model_data\addre_classes.txt'
        self.model_path = 'model_data/trained_weights_final.h5' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.005
        self.iou = 0.01
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()
        isDrawBox = True




    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def detect_image_single(self, file, isDrawBox):
        try:
            image = Image.open(file)
            width, height = image.size
        except Exception as e:
            print("Error: ", repr(e))
            return None
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('type of out_boxes: ', type(out_boxes))
        # print('type of out_scores: ', type(out_scores))
        # print('type of out_classes: ', type(out_classes))
        #
        # print(zip(out_boxes, out_scores, out_classes))
        # for i, (a, b, c) in enumerate(zip(out_boxes, out_scores, out_classes)):
        #     print(i, a, b, self.class_names[c])

        #series = pd.Series(out_boxes[:, 0])
        df = pd.DataFrame(out_boxes)
        #df = df.rename(columns={'0':'top', '1':'left', '2':'bottom','3':'right'})
        df.columns = ['top', 'left', 'bottom', 'right']
        #print(np.floor(df['top'] + 0.5))
        df['top'] = np.maximum(0, np.floor(df['top'] + 0.5)).astype('int32')
        df['left'] = np.maximum(0, np.floor(df['left'] + 0.5)).astype('int32')
        df['bottom'] = np.minimum(height, np.floor(df['bottom'] + 0.5)).astype('int32')
        df['right'] = np.minimum(width, np.floor(df['right'] + 0.5)).astype('int32')
        #print(np.minimum(0, np.floor(df['top'] + 0.5)))

        class_names = []
        for i in out_classes:
            class_names.append(self.class_names[i])
        df['class_name'] = class_names
        df['score'] = out_scores
        df['out_classes'] = out_classes
        df['image'] = os.path.basename(file)
        df['area_pct'] = (df['bottom'] - df['top']) * (df['right'] - df['left']) / (width * height)
        df['area_pct'] = df['area_pct'].abs()
        # print(df)
       # print(df['class_name'])
        if isDrawBox:
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                #
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])


                # if df.ix[i, 'top'] - label_size[1] >= 0:
                #     text_origin = np.array([df.ix[i, 'left'], df.ix[i, 'top'] - label_size[1]])
                # else:
                #     text_origin = np.array([df.ix[i, 'left'], df.ix[i, 'top'] + 1])
                #
                # # My kingdom for a good redistributable image drawing library.
                # #print(label, (left, top), (right, bottom))
                # for i in range(thickness):
                #     draw.rectangle(
                #         [df.ix[i, 'left'] + i, df.ix[i, 'top'] + i, df.ix[i, 'right'] - i, df.ix[i, 'bottom'] - i],
                #         outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        return df, image

    def detect_image_folder(self, folder):
        start = timer()
        all_files = glob.glob(folder + '/*.jpg', recursive=True)
        #df_list = [] # name, label, score, top, left, bottom, right

        print('Image number: ', len(all_files))
        print("Start detect...")
        i = 0
        p = progressbar.ProgressBar()
        p.start(len(all_files))
        w = open(os.path.join(folder,'YOLO_label_score0p005.csv'), 'w', newline="")
        w.writelines('top,left,bottom,right,class_name,score,out_classes,image,area_pct\n')
        result_folder = os.path.join(folder, 'Detected')
        for file in all_files:
            #detect_image_single(file)
            try:
                df, image = (self.detect_image_single(file, isDrawBox))
                lines = df.to_csv(header=False, index=False)
                w.writelines(lines)

                if isDrawBox:
                    if not os.path.exists(result_folder):
                        os.mkdir(result_folder)
                    #print(os.path.join(os.path.pardir(file), "Detected"))
                    image.save(os.path.join(result_folder, os.path.basename(file)))
                    #image.show()
            except Exception as e:
                print("Error in processing:", file, repr(e))
            i += 1
            p.update(i)
        w.close()

        end = timer()
        print('Processing time: %.1f' % (end - start))
        #return pd.concat(df_list)

    def close_session(self):
        self.sess.close()

def detect_img(yolo):
    while True:
        folder = input('Input image folder:')
        try:
           # image = Image.open(img)
            os.path.exists(folder)
        except:
            print('Open Folder Error! Try Again!')
            continue
        else:
            yolo.detect_image_folder(folder)
            print("Finished! ")
            #df.to_csv(, index=False)
            #r_image.show()
    yolo.close_session()



if __name__ == '__main__':
    detect_img(YOLO())
