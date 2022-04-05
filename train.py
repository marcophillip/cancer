import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from  utils import non_max_suppression
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import seaborn as sns
from data import *
# os.environ["CUDA_DEVICE_ORDER"]="10de:102d"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GeM(tf.keras.layers.Layer):
    def __init__(self,p=3.):
        super(GeM,self).__init__()
        self.epsilon = tf.keras.backend.epsilon()
        self.init_p = p

    def build(self,input_shape):
        self.shape=input_shape
        self.p = self.add_weight(initializer=tf.keras.initializers.Constant(value=self.init_p),
                                  regularizer=None,
                                  trainable=True,
                                  dtype=tf.float32)

        # print(self.shape)
        
    def call(self,inputs):
        inputs = tf.clip_by_value(inputs,self.epsilon,tf.keras.backend.max(inputs))

        avg_pool= tf.keras.layers.AveragePooling2D(pool_size=(self.shape[1],self.shape[2]))(inputs)
        avg_pool= tf.pow(avg_pool,1/self.p)
        
        return tf.reshape(avg_pool,(tf.shape(avg_pool)[0],tf.shape(avg_pool)[-1]))


class ArcFaceLayer(tf.keras.layers.Layer):
    def __init__(self,num_classes=2,margin=3,s=4):
        super(ArcFaceLayer,self).__init__()
        self.num_classes = num_classes
        self.eps=tf.keras.backend.epsilon()
        self.margin = margin
        self.s = s

    def build(self,input_shape):
        x_shape,y_shape=input_shape

        self.w = self.add_weight(
            name='arcface_weights',
            shape = (x_shape[-1],self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )
        super(ArcFaceLayer, self).build(input_shape)

    def call(self,inputs):
        x,y = inputs
        x = tf.nn.l2_normalize(x,axis=1)
        w = tf.nn.l2_normalize(self.w,axis=0)
        logits = tf.matmul(x,w)
        theta = tf.acos(tf.clip_by_value(logits,-1+self.eps,1+self.eps))
        target_logits = tf.cos(theta+self.margin)
        logits = logits*(1-y) + target_logits*y
        logits *= self.s
        return tf.nn.softmax(logits)



class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel,self).__init__()
        self.backbone=tf.keras.applications.EfficientNetB0(input_shape=(224,224,3),
                                     weights='imagenet',
                                     include_top=False)
        self.backbone.trainable=False
        self.class_input= tf.keras.layers.Input((2,))
        self.gem = GeM()
        self.glb = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.6)
        self.arcface_layer = ArcFaceLayer()
        # self.arcface_layer =tf.keras.layers.Dense(2, activation='softmax')

        self.out_ = self.call([self.backbone.input,self.class_input])

    # def build(self,input_shape):
    #     print(input_shape)

    
    def call(self,inputs):
        x,y = inputs
        x = self.backbone(x)
        x = self.glb(x)
        x = self.dropout(x)
        out =  self.arcface_layer([x,y])
        
        return out


model=CustomModel()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tb_callback=tf.keras.callbacks.TensorBoard(log_dir='/home/phillip/work/cvc2/logs')
train_data = TrainDataset(0).batch(32)
val_data = ValidationDataset(0).batch(32)


if __name__ == '__main__':
    model.fit(train_data,
            validation_data=val_data,
            epochs=1000,
            callbacks=[tb_callback]
            )