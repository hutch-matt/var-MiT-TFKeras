# Copyright 2020, MIT Lincoln Laboratory
# SPDX-License-Identifier: BSD-2-Clause

import argparse
import time
import os
import horovod.tensorflow.keras as hvd
from utils import *

if __name__ == '__main__':
    time_start = time.time()
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', action='store', dest='job_id', type=int, default=None)
    parser.add_argument('--job-name', action='store', dest='job_name', type=str, default=None)
    parser.add_argument('--batch-size-per-gpu', action='store', dest='batch_size_per_gpu', type=int, default=32)
    parser.add_argument('--reports-folder', action='store', dest='reports_folder', type=str, default=os.getcwd() + '/reports')
    parser.add_argument('--train-only', action='store_true', dest='train_only', default=False)
    parser.add_argument('--eval-only', action='store_true', dest='eval_only', default=False)
    parser.add_argument('--input-model', action='store', dest='input_model', type=str, default=None)
    parser.add_argument('--verbose', action='store', dest='verbose', type=int, default=2)
    parser.add_argument('--random-seed', action='store', dest='random_seed', type=int, default=0)
    parser.add_argument('--loss', action='store', dest='loss', type=str, default='categorical_crossentropy')
    parser.add_argument('--data-transform', action='store', dest='data_transform', type=int, default=0)
    # Required for training:
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=100)
    parser.add_argument('--initial-epoch', action='store', dest='initial_epoch', type=int, default=0)
    parser.add_argument('--training-set', action='store', dest='training_set', type=str, default='/home/gridsan/mshutch/Moments_in_Time/data-copy/data/parsed/TrainingBatch_90')
    parser.add_argument('--output-model', action='store', dest='output_model', type=str, default=None)
    parser.add_argument('--optimizer', action='store', dest='optimizer', type=str, default='sgd')
    parser.add_argument('--learning-rate', action='store', dest='learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', action='store', dest='momentum', type=float, default=0.9)
    parser.add_argument('--warmup-epochs', action='store', dest='warmup_epochs', type=int, default=5)
    parser.add_argument('--early-stopping', action='store_true', dest='early_stopping', default=False)
    # Required for validation:
    parser.add_argument('--validation-set', action='store', dest='validation_set', type=str, default='/home/gridsan/mshutch/Moments_in_Time/data-copy/data/parsed/ValidationBatch_90')
    args, unknown = parser.parse_known_args()
    
    # Prepare session
    report = args.reports_folder + '/' + args.job_name + '-report.txt'
    setup(args, report)
    args_checker(args)
    write_args(args, report)
    model = get_model(args.input_model)
    callbacks = get_callbacks(args)
    opt = get_optimizer(optimizer=args.optimizer, learning_rate=args.learning_rate, momentum=args.momentum)
    m = get_metrics()
    model.compile(optimizer=opt, loss=args.loss, metrics=m)
    
    # Training Only
    if args.train_only:
        trn_generator = DataGenerator(args, report, subset='training')
        print('Starting training on rank ' + str(hvd.rank()))
        session_start = time.time()
        # See note below about fit_generator
        hist = model.fit_generator(trn_generator, epochs=args.epochs, verbose=args.verbose if hvd.rank()==0 else 0,
                                   callbacks=callbacks)
        session_end = time.time()
        save_model(model, args.output_model, report)
        save_stats(hist, session_end-session_start, report)
    
    # Validation Only
    elif args.eval_only:
        val_generator = DataGenerator(args, report, subset='validation')
        print('Starting validation on rank ' + str(hvd.rank()))
        session_start = time.time()
        hist = model.evaluate(x=val_generator, verbose=args.verbose if hvd.rank()==0 else 0, callbacks=callbacks)
        hist = hvd.allreduce(hist)
        session_end = time.time()
        save_stats(hist, session_end-session_start, report)
        if hvd.rank()==0:
            print('All reduced stats', hist)
    
    # Training and Validation
    else:
        trn_generator = DataGenerator(args, report, subset='training')
        val_generator = DataGenerator(args, report, subset='validation')
        print('Starting training with validation on rank ' + str(hvd.rank()))
        session_start = time.time()
        # fit_generator is neccessary because fit does not yet support validation generators
        hist = model.fit_generator(trn_generator, epochs=args.epochs, verbose=args.verbose if hvd.rank()==0 else 0,
                                   callbacks=callbacks, validation_data=val_generator, validation_freq=1)
        session_end = time.time()
        save_model(model, args.output_model, report)
        save_stats(hist, session_end-session_start, report)
     
    # End session
    time_end = time.time()
    end_session(time_end-time_start, report)
    