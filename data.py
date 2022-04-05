import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from  utils import non_max_suppression
import matplotlib.pyplot as plt
import cv2
import numpy as np


classes=[]
imgs=[]
for img in os.listdir('/home/phillip/work/cvc2/data7/train/negative'):
    imgs.append(img)
    classes.append(0)
negative=pd.DataFrame({'image_paths':imgs,'classes':classes})

classes=[]
imgs=[]
for img in os.listdir('/home/phillip/work/cvc2/data7/train/positive'):
    imgs.append(img)
    classes.append(1)
positive=pd.DataFrame({'image_paths':imgs,'classes':classes})

negative['image_paths']=negative.image_paths.apply(lambda x: '/home/phillip/work/cvc2/data7/train/negative/'+x)
positive['image_paths']=positive.image_paths.apply(lambda x: '/home/phillip/work/cvc2/data7/train/positive/'+x)

train_dataframe = pd.concat([
                             negative,
                             positive])
from sklearn.utils import shuffle
train_dataframe = shuffle(train_dataframe)

train_dataframe=train_dataframe.reset_index().drop('index',axis=1)

skf = StratifiedKFold(n_splits=5)
train_dataframe['fold'] = -1
for i, (train_idx,val_idx) in enumerate(skf.split(train_dataframe,train_dataframe.classes.values.tolist())):
    train_dataframe.loc[val_idx,'fold']=i
                                        


# localization_model = tf.keras.models.load_model('/home/phillip/work/yolov5/runs/train/exp23/weights/last_saved_model')

class TrainDataset(tf.data.Dataset):

    def _generator(fold):
        train = train_dataframe[train_dataframe.fold != fold]
        paths = train.image_paths.values.tolist()
        classes = train.classes.values.tolist()
        pred_boxes = []
        true_boxes = []
        i = 0
        while i < len(paths):
            try:
                img = cv2.imread(paths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img_416 = cv2.resize(img, (416, 416)) / 255.
                # b = localization_model.predict(np.expand_dims(img_416, 0))
                # v = non_max_suppression(b)
                # xyxy = (v[0][:4] * 1024).astype('int')

                # img_cropped = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                img_224 = cv2.resize(img, (224, 224))

                # lab = cv2.cvtColor(img_224, cv2.COLOR_RGB2Lab)
                # indx_ = np.where(lab[:, :, 0] > [180])


                # for ii in range(len(indx_[0])):
                #     i1 = img_224[abs(indx_[0][ii] - 2):abs(indx_[0][ii] + 2),
                #          abs(indx_[1][ii] - 2):abs(indx_[1][ii] + 2)]
                #     img_224[indx_[0][ii], indx_[1][ii]] = i1.mean((0, 1)).astype('float32')

                img_224 = tf.keras.applications.efficientnet.preprocess_input(img_224)
                # img_224 = tf.expand_dims(img_224,axis=0)
                yield (img_224, tf.keras.utils.to_categorical(np.array(classes[i]), 2)), tf.keras.utils.to_categorical(
                    np.array(classes[i]), 2)
                i += 1
            except:
                i += 1

    def __new__(cls, fold):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.float32, tf.float32), tf.float32),
            output_shapes=(((224, 224, 3), (2)), (2)),
            args=(fold,)

        )


class ValidationDataset(tf.data.Dataset):

    def _generator(fold):
        train = train_dataframe[train_dataframe.fold == fold]
        paths = train.image_paths.values.tolist()
        classes = train.classes.values.tolist()
        pred_boxes = []
        true_boxes = []
        i = 0
        while i < len(paths):
            try:
                img = cv2.imread(paths[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img_416 = cv2.resize(img, (416, 416)) / 255.
                # b = localization_model.predict(np.expand_dims(img_416, 0))
                # v = non_max_suppression(b)
                # xyxy = (v[0][:4] * 1024).astype('int')

                # img_cropped = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                img_224 = cv2.resize(img, (224, 224))

                # lab = cv2.cvtColor(img_224, cv2.COLOR_RGB2Lab)
                # indx_ = np.where(lab[:, :, 0] > [180])
    

                # for ii in range(len(indx_[0])):
                #     i1 = img_224[abs(indx_[0][ii] - 2):abs(indx_[0][ii] + 2),
                #          abs(indx_[1][ii] - 2):abs(indx_[1][ii] + 2)]
                #     img_224[indx_[0][ii], indx_[1][ii]] = i1.mean((0, 1)).astype('int')

                img_224 = tf.keras.applications.efficientnet.preprocess_input(img_224)

                yield (img_224, tf.keras.utils.to_categorical(np.array(classes[i]), 2)), tf.keras.utils.to_categorical(
                    np.array(classes[i]), 2)
                i += 1
            except:
                i += 1

    def __new__(cls, fold):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.float32, tf.float32), tf.float32),
            output_shapes=(((224, 224, 3), (2)), (2)),
            args=(fold,)

        )