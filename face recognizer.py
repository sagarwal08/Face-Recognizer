import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.layers import Dense , Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Dropout,  Flatten, MaxPooling2D, Conv2D , BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split as tts
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def  aithon_level2_api(traingcsv, testcsv):


    data = pd.read_csv(traingcsv)#aithon2020_level2_traning.csv
    test = pd.read_csv(testcsv)

    data = pd.DataFrame(data)
    print(data.shape)
    data.replace({'Fear': 0, 'Sad':1, 'Happy':2}, inplace=True)

    test['emotion']=test.emotion.astype(object)

    test = pd.DataFrame(test)
    print(test.shape)
    test.replace({'Fear': 0, 'Sad':1, 'Happy':2}, inplace=True)

    

    Y = data['emotion']
    x = data.drop('emotion', axis=1)
    x = x.values
    Y = Y.values

    Ytest = test['emotion']
    xtest = test.drop('emotion', axis=1)
    xtest = xtest.values
    Ytest = Ytest.values
    #xtest.drop[columns='']
    

    x = x.reshape((data.shape[0],48,48,1))

    xtest = xtest.reshape((test.shape[0],48,48,1))

    y = np.zeros((Y.size, Y.max()+1))
    y[np.arange(Y.size),Y] = 1

    ytest = np.zeros((Ytest.size, Ytest.max()+1))
    ytest[np.arange(Y.size),Y] = 1

    print(y.shape)
    print(x.shape)

    print(ytest.shape)
    print(xtest.shape)

    xtrain, xval, ytrain, yval = tts(x, y, train_size=0.98, stratify=y, random_state=0)
    #xtest, xval, ytest, yval = tts(xtestval, ytestval, train_size=0.5, random_state=44)

    xtrain = xtrain.reshape(xtrain.shape[0],48,48,1)
    #xtest = xtest.reshape(108,48,48,1)
    xval = xval.reshape(xtest.shape[0],48,48,1)

    xtrain = xtrain/255
    #xtest = xtest/255
    xval = xval/255

    datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  

    datagen.fit(xtrain)

    def callbacks(pat1, pat2):
        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=pat1, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=pat2, restore_best_weights=True)


        return [lr_reduce, early_stopping]


    def swish(x):
        return (K.sigmoid(x) * x)

    get_custom_objects().update({'swish': swish})



    image = Input(shape=(48, 48, 1))
    l = Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(48,48,1), kernel_initializer="he_normal")(image)
    l = Conv2D(32, (3, 3), padding="same", activation='relu', kernel_initializer="he_normal")(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = BatchNormalization()(l)
    l = Conv2D(64, (3, 3), activation='relu', padding="same")(l)
    l = Conv2D(64, (3, 3), padding="same", activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = BatchNormalization()(l)
    l = Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same")(l)
    l = Conv2D(96, (3, 3), padding="valid", activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = BatchNormalization()(l)
    l = Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same")(l)
    l = Dropout(0.4)(l)
    l = Conv2D(128, (3, 3), padding="valid", activation='relu')(l)
    l = MaxPooling2D(pool_size=(2, 2))(l)
    l = BatchNormalization()(l)
    l = Flatten()(l)
    l = Dense(64, activation=swish)(l)
    l = Dropout(0.4)(l)
    l = Dense(3 , activation='sigmoid')(l)

    model = Model(inputs = image, outputs = l)

    #pls see for the location here--------------------------------------------------------
    model.load_weights(r'FER_weights.h5')
    #-------------------------------------------------------------------------------------
    print(model.summary())

    
    callback = callbacks(6,12)
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    batch_size = 128
    history = model.fit(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                        steps_per_epoch= xtrain.shape[0]//batch_size,
                        callbacks=callback,
                        validation_data=(xval, yval),
                        epochs = 50)
    
    
    callback = callbacks(4,10)
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    batch_size = 128
    history = model.fit(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                        steps_per_epoch= xtrain.shape[0]//batch_size,
                        callbacks=callback,
                        validation_data=(xval, yval),
                        epochs = 30)
    
    callback = callbacks(3,8)
    adam = tf.keras.optimizers.Adam(lr=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    batch_size = 512
    history = model.fit(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                        steps_per_epoch= xtrain.shape[0]//batch_size,
                        callbacks=callback,
                        validation_data=(xval, yval),
                        epochs = 25)
    
    callback = callbacks(3,8)
    adam = tf.keras.optimizers.Adam(lr=0.000001)
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    batch_size = 1024
    history = model.fit(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                        steps_per_epoch= xtrain.shape[0]//batch_size,
                        callbacks=callback,
                        validation_data=(xval, yval),
                        epochs = 15)
    
    callback = callbacks(2,5)
    adam = tf.keras.optimizers.Adam(lr=0.0000001)
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    batch_size = 1024
    history = model.fit(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                        steps_per_epoch= xtrain.shape[0]//batch_size,
                        callbacks=callback,
                        validation_data=(xval, yval),
                        epochs = 15)
    
    callback = callbacks(2,6)
    adam = tf.keras.optimizers.Adam(lr=0.00000003)
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    batch_size = 1024
    history = model.fit(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                        steps_per_epoch= xtrain.shape[0]//batch_size,
                        callbacks=callback,
                        validation_data=(xval, yval),
                        epochs = 15)
    
    
    
    
    #now the final model has been trained!! can use model.predict on test data
    y_test_pred=model.predict(xtest)
    print("y_testpredict: ",y_test_pred.shape)
    y_output=[]
    for i in y_test_pred:
        if y_test_pred==0:
            y_output.append('Fear')
        elif y_test_pred==1:
           y_output.append("Sad")
        else:
            y_output.append("Happy")
    
    y_test_pred_ = []
    for i in range(y_test_pred.shape[0]):
        y_test_pred_.append(y_test_pred[i].argmax())
    print(y_test_pred_)
    y_output=[]
    for i in range(len(y_test_pred_)):
        if y_test_pred_[i]==0:
            y_output.append('Fear')
        elif y_test_pred_[i]==1:
            y_output.append("Sad")
        else:
            y_output.append("Happy")

    print(len(y_output))
    print(y_output)

    return y_output

    
    
#y_test_pred_ = np.array(y_test_pred)


aithon_level2_api('aithon2020_level2_traning.csv', 'Name.csv')