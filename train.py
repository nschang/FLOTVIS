import keras.backend as K
import numpy as np
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from nets.loss import yolo_loss
from nets.yolo4 import yolo_body
from utils.utils import (LossHistory, WarmUpCosineDecayScheduler,
                         get_random_data, get_random_data_with_Mosaic, get_classes, get_anchors)
from utils.utils_augmentation import YoloDatasets, preprocess_true_boxes

# ------------------------------
# train data generator
# ------------------------------
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False, random=True):
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i+4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i+4], input_shape)
                    i = (i+4) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                    i = (i+1) % n
                flag = bool(1-flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                i = (i+1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

# ------------------------------
if __name__ == "__main__":
    annotation_path = '2007_train.txt'
    log_dir = 'train/'
    # ------------------------------
    # set training parameters
    # ------------------------------
    classes_path = 'model_data/class.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    weights_path = 'model_data/yolo4_weight.h5'
    # ------------------------------
    # img size for training
    # ------------------------------
    input_shape = (608,608)
    # ------------------------------
    # set batch normalization
    # ------------------------------
    normalize = True
    # ------------------------------
    # count classes and anchors
    # ------------------------------
    class_names = get_classes(classes_path)
    anchors     = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # ------------------------------
    # Yolov4 tricks
    # ------------------------------
    mosaic = False
    Cosine_scheduler = True
    label_smoothing = 0.01

    K.clear_session()
    # ------------------------------
    # create yolo model
    # ------------------------------
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    
    # ------------------------------
    # load pre-trained weight
    # ------------------------------
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    
    # ------------------------------
    # set loss
    # ------------------------------
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
    loss_input = [*model_body.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 
            'label_smoothing': label_smoothing, 'normalize': normalize})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)
    loss_history = LossHistory(log_dir)

    # ------------------------------
    # training set and validation set ratio to 1:9
    # ------------------------------
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    freeze_layers = 249
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    if True:
        Init_epoch          = 0
        Freeze_epoch        = 50
        batch_size          = 16
        learning_rate_base  = 1e-3

        if Cosine_scheduler:
            # warmup
            warmup_epoch    = int((Freeze_epoch-Init_epoch)*0.2)
            # total steps
            total_steps     = int((Freeze_epoch-Init_epoch) * num_train / batch_size)
            # warmup steps
            warmup_steps    = int(warmup_epoch * num_train / batch_size)
            # learning rate
            reduce_lr       = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                        total_steps=total_steps,
                                                        warmup_learning_rate=1e-4,
                                                        warmup_steps=warmup_steps,
                                                        hold_base_rate_steps=num_train,
                                                        min_learn_rate=1e-6
                                                        )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        epoch_size          = num_train // batch_size
        epoch_size_val      = num_val // batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("Dataset is too small to train. Please get more data.")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic, random=True),
                steps_per_epoch=epoch_size,
                validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False, random=False),
                validation_steps=epoch_size_val,
                epochs=Freeze_epoch,
                initial_epoch=Init_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    for i in range(freeze_layers): model_body.layers[i].trainable = True

    if True:
        Freeze_epoch        = 50
        Epoch               = 100
        batch_size          = 4
        learning_rate_base  = 1e-4

        if Cosine_scheduler:
            # warmup
            warmup_epoch = int((Epoch-Freeze_epoch)*0.2)
            # total steps
            total_steps = int((Epoch-Freeze_epoch) * num_train / batch_size)
            # warmup steps
            warmup_steps = int(warmup_epoch * num_train / batch_size)
            # learning rate
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                        total_steps=total_steps,
                                                        warmup_learning_rate=1e-5,
                                                        warmup_steps=warmup_steps,
                                                        hold_base_rate_steps=num_train//2,
                                                        min_learn_rate=1e-6
                                                        )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        epoch_size          = num_train // batch_size
        epoch_size_val      = num_val // batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("Dataset is too small to train. Please get more data.")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic, random=True),
                steps_per_epoch=epoch_size,
                validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False, random=False),
                validation_steps=epoch_size_val,
                epochs=Epoch,
                initial_epoch=Freeze_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])
        model.save_weights(log_dir + 'last1.h5')
