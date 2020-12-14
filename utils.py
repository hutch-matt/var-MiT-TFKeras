# Copyright 2020, MIT Lincoln Laboratory
# SPDX-License-Identifier: BSD-2-Clause

import os
import argparse
import time
import horovod.tensorflow.keras as hvd
import numpy as np
import tensorflow as tf
import h5py
import math
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Adadelta
from tensorflow.python.keras.utils import to_categorical, Sequence
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks as cb

def read_as_list(filename):
    """Read the file at filename and store its contents into a list."""
    with open(filename) as f:
        result = [line.rstrip() for line in f.readlines()]
        f.close()
        return result

def setup(args, report):
    """Set up environment variables given the type of partition."""
    # Initialize Horovod
    hvd.init()
    # Set environment variable necessary to use h5py for file read/write
    os.putenv("HDF5_USE_FILE_LOCKING", "FALSE")
    os.system("export $HDF5_USE_FILE_LOCKING")
    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.compat.v1.Session(config=config))
    np.random.seed(args.random_seed)
    print('Rank ' + str(hvd.rank()) + ' session configured')
    
def args_checker(args):
    """Verify no argument conflicts or issues."""
    assert args.job_id is not None
    assert args.job_name is not None
    assert args.batch_size_per_gpu >= 1
    assert args.reports_folder is not None
    assert not (args.train_only and args.eval_only)
    assert args.input_model is not None
    assert args.verbose in [0,1,2]
    assert args.random_seed >= 0
    assert args.loss is not None
    # The following are needed to compile the model even if no training occurs
    assert args.optimizer is not None
    assert args.learning_rate is not None
    assert args.momentum is not None
    assert args.data_transform is not None
    if not args.eval_only:
        assert args.epochs >= 1
        assert args.initial_epoch >= 0
        assert args.training_set is not None
        assert args.output_model is not None
        assert args.optimizer is not None
        assert args.learning_rate > 0
        assert args.momentum >= 0
        assert args.warmup_epochs >= 0
    if not args.train_only:
        assert args.validation_set is not None
    
def write_args(args, report):
    """Write the arguments to the report file."""
    if hvd.rank() == 0:
        # Write to log
        print('Arguments', str(args))
        # Write to report
        r = open(report, 'a+')
        r.write('----------------------------------------------------\n')
        if args.train_only:
            r.write('Training session:\n')
        elif args.eval_only:
            r.write('Validation session:\n')
        else:
            r.write('Training and Validation session:\n')
        r.write(str(args) + '\n')
        r.close()
        
def get_model(model_file, log=True):
    """Load a model from the specified model_file."""
    model = load_model(model_file)
    if log:
        print('Model successfully loaded on rank ' + str(hvd.rank()))
    return model

def get_callbacks(args):
    """Define callbacks for distributed training."""
    callbacks = [
            # This is necessary to ensure consistent initialization of all workers
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            # Note: must be in the list before the ReduceLROnPlateau or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),
            # Adjust Learning Rate
            hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs)
            ]    
    if args.train_only:
        # Reduce learning rate on a schedule
        onethirds_point = int(math.floor(args.epochs / 3))
        twothirds_point = int(math.floor(args.epochs / 3 * 2))
        callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=onethirds_point, multiplier=1.))
        callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=onethirds_point, end_epoch=twothirds_point, multiplier=1e-1))
        callbacks.append(hvd.callbacks.LearningRateScheduleCallback(start_epoch=twothirds_point, end_epoch=args.epochs+1, multiplier=1e-2))
    else:
        # Reduce learning rate on validation loss plateau
        callbacks.append(cb.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001, verbose=1 if hvd.rank()==0 else 0))
    if args.early_stopping:
        callbacks.append(cb.EarlyStopping(monitor='loss', patience=7, restore_best_weights=True))
    print('Callbacks created on rank ' + str(hvd.rank()))
    return callbacks

def get_optimizer(optimizer='sgd', learning_rate=0.1, momentum=0.9, log=True):
    """Create an optimizer and wrap it for Horovod distributed training. Default is SGD."""
    if log:
        print('Creating optimizer on rank ' + str(hvd.rank()))
    opt = None
    if optimizer == 'sgd+nesterov':
        opt = SGD(lr=learning_rate, momentum=momentum, nesterov=True)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=learning_rate, rho=0.9)
    elif optimizer == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=learning_rate, rho=0.95)
    else:
        opt = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    # Wrap optimizer for data distributed training
    return hvd.DistributedOptimizer(opt)

class DataGenerator(tf.compat.v2.keras.utils.Sequence):    
    def __init__(self, args, report, subset='training'):
        # Initialize data generator variables
        self.args = args
        self.report = report
        self.subset = subset
        self.data = get_data_pointers(self.args, self.report, self.subset)
        self.shuffle = True if subset=='training' else False
        self.n = 0
        # Shuffle data for training before the first epoch
        if self.shuffle:
            self.on_epoch_end()
            
    def __next__(self):
        # Return the next data batch
        data = self.__getitem__(self.n)
        self.n += 1
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0
        return data
    
    def __len__(self):
        # Return the number of batches of the dataset
        total_pts = len(self.data)
        pts_on_this_rank = math.ceil(total_pts / hvd.size())
        return math.floor(pts_on_this_rank / self.args.batch_size_per_gpu)
    
    def __getitem__(self, index):
        # Generate the data batch at a given index
        start = hvd.rank() * math.ceil(len(self.data)/hvd.size()) + self.args.batch_size_per_gpu * index
        end = min(start + self.args.batch_size_per_gpu, len(self.data))
        batch_data = self.data[start:end]
        X = []
        y = []
        for datum in batch_data:
            arr, label = transform(datum, transform_num=self.args.data_transform)
            X.append(arr)
            y.append(label)
        return np.array(X), np.array(y)
    
    def on_epoch_end(self):
        # Shuffle data for training at the end of the epoch
        if self.shuffle:
            if hvd.rank()==0:
                print('Shuffling data for new epoch')
            np.random.shuffle(self.data) 
            
def save_model(model, output_file, report):
    """Save the fully trained model."""
    if hvd.rank() == 0:
        model.save(output_file)
        r = open(report, 'a+')
        r.write('Trained model saved to ' + output_file + '\n')
        r.close()
            
def save_stats(hist, total_time, report):
    """Save session statistics."""
    if hvd.rank() == 0:
        r = open(report, 'a+')
        r.write(str(hist))
        try:
            r.write(str(hist.history))
        except:
            pass
        r.write('\n')
        r.write('total_session_time ' + str(total_time) + ' seconds \n')
        r.close()
        
def end_session(full_time, report):
    """End session and log final time."""
    print('Total session time on rank', hvd.rank(), 'is', full_time, 'seconds')
    if hvd.rank() == 0:
        r = open(report, 'a+')
        r.write('full_start_to_end_time ' + str(full_time) + ' seconds \n')
        r.close()
        
            
###################################################################
# Methods to modify for your specific problem
# Examples shown are for video action recognition

def get_metrics():
    """Create metrics for the model."""
    metrics = []
    # Loss is included by default
    # Top-1 Accuracy
    metrics.append('acc')
    # Top-5 Accuracy
    metrics.append(tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top-'+str(5)+'-acc'))
    # Add other metrics here add desired
    return metrics

def get_data_pointers(args, report, subset):
    """Creates a pointer to each data point in the dataset."""
    data_tuples = []
    set_dir = None
    if subset=='training':
        set_dir = args.training_set
    else:
        set_dir = args.validation_set
    # Read folders in set directory. Folders should correspond to classes in this case
    categories = os.listdir(set_dir)
    # Remove any extraneous .ipynb files
    new_categories = []
    for cat in categories:
        if '.ipynb' not in cat:
            new_categories.append(cat)
    categories = new_categories
    # Create one-hot vector labels
    categories_labels = dict()
    for i in range(len(categories)):
        categories_labels[categories[i]] = to_categorical(np.array(i), num_classes=len(categories))
    # Create data pointer tuples (file, datapoint, label)
    for cat in categories_labels.keys():
        cat_dir = set_dir + '/' + cat
        data_file = h5py.File(cat_dir + '/' + cat + '-videos.hdf5')
        keys = read_as_list(cat_dir + '/' + 'video-keys.txt')
        label = categories_labels[cat]
        for key in keys:
            data_tuples.append( (data_file, key, label) )
    if hvd.rank()==0:
        print('Found ' + str(len(data_tuples)) + ' data points from ' + str(len(categories_labels)) + ' categories')
    return data_tuples  
            
def transform(datum, transform_num=0):
    """Extract and transform data as needed."""
    data_file, key, label = datum
    arr = data_file[key]
    assert len(arr) > 0
    if transform_num==0:
        # Default, no transformation, return data "as is"
        return arr, label
    else:
        if transform_num==1:
            # Sample a frame uniformly at random [f,c,h,w] --> [c,h,w]   
            frame_num = np.random.randint(len(arr))
            arr = arr[frame_num,:,:,:]
            # Reorder so channels are last [c,h,w] --> [h,w,c]
            arr = channels_first_to_last(arr)
            # Random crop [h,w,c] --> [h',w',c]
            arr = random_crop(arr, new_h=224, new_w=224)
            # Random horizontal flip [h,w,c] --> [h,w,c]
            arr = random_flip(arr)
            return arr, label
        if transform_num==2:
            # Loop video as needed until frames > 16 [f,c,h,w] --> [f',c,h,w]
            while len(arr) < 16:
                arr = np.concatenate((arr, arr), axis=0)
            # Sample 16 dense frames uniformly at random [f,c,h,w] --> [f',c,h,w]
            frame_start = np.random.randint(len(arr)-16)
            frame_end = frame_start + 16
            arr = arr[frame_start:frame_end,:,:,:]
            # Reorder so channels are last [f,c,h,w] --> [f,h,w,c]
            arr = channels_first_to_last_3D(arr)
            # Random crop [f,h,w,c] --> [f,h',w',c]
            arr = random_crop_3D(arr, new_h=224, new_w=224)
            # Random horizontal flip [f,h,w,c] --> [f,h,w,c]
            arr = random_flip_3D(arr)
            return arr, label
        if transform_num==3:
            # Loop video as needed until frames > 64 [f,c,h,w] --> [f',c,h,w]
            while len(arr) < 64:
                arr = np.concatenate((arr, arr), axis=0)
            # Sample 64 dense frames uniformly at random [f,c,h,w] --> [f',c,h,w]
            frame_start = np.random.randint(len(arr)-64)
            frame_end = frame_start + 64
            arr = arr[frame_start:frame_end,:,:,:]
            # Reorder so channels are last [f,c,h,w] --> [f,h,w,c]
            arr = channels_first_to_last_3D(arr)
            # Random crop [f,h,w,c] --> [f,h',w',c]
            arr = random_crop_3D(arr, new_h=224, new_w=224)
            # Random horizontal flip [f,h,w,c] --> [f,h,w,c]
            arr = random_flip_3D(arr)
            return arr, label
        if transform_num==4:
            # Loop video as needed until frames > 16 [f,c,h,w] --> [f',c,h,w]
            while len(arr) < 16:
                arr = np.concatenate((arr, arr), axis=0)
            # Define segment length
            seg_len = math.floor(len(arr)/16)
            # Sample 16 spread frames uniformly at random [f,c,h,w] --> [f',c,h,w]
            frame_start = np.random.randint(seg_len)
            arr = arr[[frame_start,frame_start+1*seg_len,frame_start+2*seg_len,frame_start+3*seg_len,frame_start+4*seg_len,frame_start+5*seg_len,frame_start+6*seg_len,frame_start+7*seg_len,frame_start+8*seg_len,frame_start+9*seg_len,frame_start+10*seg_len,frame_start+11*seg_len,frame_start+12*seg_len,frame_start+13*seg_len,frame_start+14*seg_len,frame_start+15*seg_len],:,:,:]
            # Reorder so channels are last [f,c,h,w] --> [f,h,w,c]
            arr = channels_first_to_last_3D(arr)
            # Random crop [f,h,w,c] --> [f,h',w',c]
            arr = random_crop_3D(arr, new_h=224, new_w=224)
            # Random horizontal flip [f,h,w,c] --> [f,h,w,c]
            arr = random_flip_3D(arr)
            return arr, label
        if transform_num==5:
            # Loop video as needed until frames > 32 [f,c,h,w] --> [f',c,h,w]
            while len(arr) < 32:
                arr = np.concatenate((arr, arr), axis=0)
            # Sample 32 dense frames uniformly at random [f,c,h,w] --> [f',c,h,w]
            frame_start = np.random.randint(len(arr)-32)
            frame_end = frame_start + 32
            arr = arr[frame_start:frame_end,:,:,:]
            # Reorder so channels are last [f,c,h,w] --> [f,h,w,c]
            arr = channels_first_to_last_3D(arr)
            # Random crop [f,h,w,c] --> [f,h',w',c]
            arr = random_crop_3D(arr, new_h=224, new_w=224)
            # Random horizontal flip [f,h,w,c] --> [f,h,w,c]
            arr = random_flip_3D(arr)
            return arr, label
        # Add new data_transformation sequences here
        print('Should never reach here')
        return None, None
        
def channels_first_to_last(arr):
    """Swap array of shape (channels, dim, dim) to (dim, dim, channels)."""
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr
       
def channels_first_to_last_3D(arr):
    """Swap array of shape (frames, channels, dim, dim) to (frames, dim, dim, channels)."""
    arr = np.swapaxes(arr, 1, 2)
    arr = np.swapaxes(arr, 2, 3)
    return arr    

def random_crop(arr, new_h=224, new_w=224):
    """Crop an image of shape (dim, dim, channels) to (new_h, new_w, channels)."""
    height = len(arr)
    width = len(arr[0])
    assert height >= new_h
    assert width >= new_w
    if height > new_h or width > new_w:
        height_sample_pt = np.random.randint(height-new_h)
        width_sample_pt = np.random.randint(width-new_w)
        return arr[height_sample_pt:height_sample_pt+new_h,width_sample_pt:width_sample_pt+new_w,:]
    else:
        return arr

def random_crop_3D(arr, new_h=224, new_w=224):
    """Crop a video of shape (frames, dim, dim, channels) to (frames, new_h, new_w, channels)."""
    frame = arr[0]
    height = len(frame)
    width = len(frame[0])
    assert height >= new_h
    assert width >= new_w
    if height > new_h or width > new_w:
        height_sample_pt = 0
        width_sample_pt = 0
        if height > new_h:
            height_sample_pt = np.random.randint(height-new_h)
        if width > new_w:
            width_sample_pt = np.random.randint(width-new_w)
        return arr[:,height_sample_pt:height_sample_pt+new_h,width_sample_pt:width_sample_pt+new_w,:]
    else:
        return arr
    
def random_flip(arr):
    """Flip an image along its horizontal direction."""
    if np.random.rand() > 0.5:
        return np.flip(arr, 1)
    else: 
        return arr

def random_flip_3D(arr):
    """Flip a video along its horizontal direction."""
    if np.random.rand() > 0.5:
        return np.flip(arr, 2) 
    else:
        return arr
