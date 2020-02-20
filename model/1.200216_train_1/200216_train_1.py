from absl import app, flags, logging
import numpy as np
import pandas as pd
from numpy.random import rand
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Input, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.callbacks import ModelCheckpoint
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import pprint
import argparse
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--mode', 
                        default=1, 
                        type=int,
                        help='mode1 :real_mode, mode2: test_mode')
    parser.add_argument('--train_folder', 
                        default = '../data/9.split_train,test/train', 
                        type=str,
                        help='path to the binary image file')
    parser.add_argument('--test_folder', 
                        default='../data/9.split_train,test/test', 
                        type=str,
                        help='path to the image index path')
    parser.add_argument('--base_modelname', 
                        default='inceptionV3', 
                        type=str,
                        help='base_model')                    
    parser.add_argument('--COUNT', 
                        default = 1 ,
                        type = int,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--BATCH_SIZE', 
                        default = 32, 
                        type=int,
                        help='batch_size')
    parser.add_argument('--EPOCH', 
                        default=15, 
                        type=int,
                        help='number of epoch')
    parser.add_argument('--IMG_HEIGHT', default=299, type=int,
                        help='img_height')
    parser.add_argument('--IMG_WIDTH', default=299, type=int,
                        help='img_width')
    args = parser.parse_args()
    if args.mode == 2:
        args = args_test(args)
    print('args:', args)
    return args


def args_test(args):
    logging.info('test mode on lacal')
    args.train_folder = '../data/dog_9/train'
    args.base_modelname = 'basic' 
    args.test_folder = '../data/dog_9/test'
    args.COUNT = 1
    args.BATCH_SIZE = 32
    args.EPOCH = 1
    args.IMG_HEIGHT= 299
    args.IMG_WIDTH = 299
    return args


def img_generator(args):
    logging.info(' Create train generator.')
    train_datagen = ImageDataGenerator(#rescale=1./255, 
                                    preprocessing_function = preprocess_input,
                                    rotation_range=30, 
                                    width_shift_range=0.2,
                                    height_shift_range=0.2, 
                                    validation_split =0.15,
                                    horizontal_flip = 'true')
    train_generator = train_datagen.flow_from_directory(directory= args.train_folder,
                                                        shuffle=True,
                                                        target_size = (args.IMG_HEIGHT, args.IMG_WIDTH),
                                                        batch_size=args.BATCH_SIZE,
                                                        subset = 'training',
                                                        seed=1,
                                                        interpolation='nearest')
    val_generator = train_datagen.flow_from_directory(directory= args.train_folder,
                                                        shuffle=True,
                                                        target_size = (args.IMG_HEIGHT, args.IMG_WIDTH),
                                                        batch_size=args.BATCH_SIZE,
                                                        subset = 'validation',
                                                        seed=1,
                                                        interpolation='nearest')

    return train_generator, val_generator



def get_model(args):
    logging.info('get model')
    # Get the InceptionV3 model so we can do transfer learning
    if args.base_modelname == 'inceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top = False, input_shape=(299, 299, 3))
        out = base_model.output
        out = Flatten()(out)
        # out = GlobalAveragePooling2D()(out)
        out = Dense(512, activation='relu')(out)
        out = Dense(512, activation='relu')(out)
        total_classes = train_generator.num_classes
        predictions = Dense(total_classes, activation='softmax')(out)
        model = Model(inputs=base_model.input, outputs=predictions)



        
    elif args.base_modelname == 'inceptionResNetV2':    
        base_model = InceptionResNetV2(weights='imagenet', include_top = False, input_shape=(args.IMG_WIDTH,args.IMG_WIDTH,3))
        out = base_model.output
        out = GlobalAveragePooling2D()(out)
        out = Dense(512, activation='relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(512, activation='relu')(out)
        out = Dropout(0.5)(out)
        total_classes = train_generator.num_classes
        predictions = Dense(total_classes, activation='softmax')(out)
        model = Model(inputs=base_model.input, outputs=predictions)


    elif args.base_modelname == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top = False, input_shape=(args.IMG_WIDTH,args.IMG_WIDTH,3))
        out = base_model.output
        out = GlobalAveragePooling2D()(out)
        out = Dense(512, activation='relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(512, activation='relu')(out)
        out = Dropout(0.5)(out)
        total_classes = train_generator.num_classes
        predictions = Dense(total_classes, activation='softmax')(out)
        model = Model(inputs=base_model.input, outputs=predictions)
           
    else:
        x= Input(shape=(args.IMG_WIDTH,args.IMG_WIDTH,3))
        out = GlobalAveragePooling2D()(x)
        out = Dense(128, activation='relu')(out)
        out = Dropout(0.5)(out)
        total_classes = train_generator.num_classes
        predictions = Dense(total_classes, activation='softmax')(out)
        model = Model(inputs=x, outputs=predictions)


    model.compile(Adam(lr = .0001), 
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

    # model.summary()
    return model


def make_dir(args):
    logging.info('make dir for model saving')
    now = datetime.now()
    model_path = '../model_save/'
    dir_name = now.strftime('%y%m%d')+'_' + str(args.COUNT)
    dir_path = os.path.join(model_path, dir_name)
    os.makedirs(dir_path, exist_ok = True)
    save_model_path= dir_path
    return save_model_path, dir_name


def model_train(save_model_path, dir_name, args):
    logging.info('model training')
    batch_size = args.BATCH_SIZE
    train_steps_per_epoch = train_generator.n // batch_size
    val_steps_per_epoch = val_generator.n // batch_size
    dir_path = save_model_path

    # checkpoint
    checkpoint = ModelCheckpoint( dir_path + '/' + dir_name + "_{epoch:02d}_{val_acc:.4f}.hdf5", 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, # 덮어쓰기
                                 save_weights_only = True,  # 가중치만 저장
                                 mode='auto')

    

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_steps_per_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_steps_per_epoch,
                                  epochs=args.EPOCH,
                                  verbose=0,
                                  callbacks = [checkpoint])
    print('model train END!')
    return history, model


def visualize_model_perfomance(history, save_model_path, dir_name, args):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(args.EPOCH, 5))
    t = f.suptitle('Deep Neural Net Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epochs = list(range(1,args.EPOCH+1))
    ax1.plot(epochs, history.history['acc'], label='Train Accuracy')
    ax1.plot(epochs, history.history['val_acc'], label='Validation Accuracy')
    ax1.set_xticks(epochs)
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epochs, history.history['loss'], label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(epochs)
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    plt.savefig(save_model_path + '/' + dir_name+".png", pad_inches = 0, dpi = 150)
    logging.info('save model history plot')

def test_model_performance(args):
    logging.info('model evaluating')

    test_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
    )
    test_generator = test_generator.flow_from_directory(batch_size = args.BATCH_SIZE,
                                                       directory = args.test_folder,
                                                       target_size = (args.IMG_HEIGHT, args.IMG_WIDTH),
                                                       )

    STEP_SIZE_TEST = test_generator.n/args.BATCH_SIZE

    scores = model.evaluate_generator(test_generator,
                                      steps = STEP_SIZE_TEST)

    print('%s: %.2f%%' %(model.metrics_names[1], scores[1] * 100))
    return test_generator


def confusion_matrix_report(test_genrator, target_names,save_model_path,args):
    steps = test_generator.n // args.BATCH_SIZE 

    Y_pred = model.predict_generator(test_generator,
                                    # steps
                                    )
    y_pred = np.argmax(Y_pred, axis = 1)
    conf_mat = confusion_matrix(test_generator.classes, y_pred)
    report = classification_report(test_generator.classes, y_pred, target_names = target_names)
    # df = pd.DataFrame(report).transpose()
    # df.to_save('../save_model_path/report.csv')
    # logging.info('save report.csv')
  
    print(report)
    return report, test_genrator.classes, y_pred



def score_df(true, pred, target_names,save_model_path, dir_name):
    clf_rep = metrics.precision_recall_fscore_support(true, pred)
    out_dict = {
                "precision" :clf_rep[0].round(2)
                ,"recall" : clf_rep[1].round(2)
                ,"f1-score" : clf_rep[2].round(2)
                ,"support" : clf_rep[3]
                }
    out_df = pd.DataFrame(out_dict, index = target_names)
    avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name!="support" else  round(x.sum(), 2)).to_frame().T)
    avg_tot.index = ["avg/total"]
    out_df = out_df.append(avg_tot)
    out_df.to_csv(save_model_path+'/'+ dir_name + '.matrix_report.csv')
    print(out_df)
    return out_df


if __name__=='__main__':
    args = parse_args()
    train_generator, val_generator = img_generator(args)
    target_names = list(train_generator.class_indices.keys())
    model = get_model(args)
    save_model_path, dir_name= make_dir(args)
    history, model = model_train(save_model_path, dir_name, args)
    visualize_model_perfomance(history,save_model_path, dir_name, args)
    test_generator = test_model_performance(args)
    report, true, pred = confusion_matrix_report(test_generator, target_names, save_model_path, args)
    out_df = score_df(true, pred, target_names, save_model_path,dir_name)