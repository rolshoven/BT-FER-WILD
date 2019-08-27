from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from generator import TrainDataGenerator, ValDataGenerator
import os


class EpochLRDiscovery(Callback):
    """
    Searches for good learning rates at the beginning of every epoch. A learning rate is considered good if it yields
    the best validation accuracy.
    """
    def __init__(self,
                 start_lr,
                 min_lr,
                 max_lr,
                 num_lr=10,
                 neighborhood_log_range=1.5,
                 neighborhood_fraction=0.4,
                 epoch_range_factor=0.8,
                 num_samples=10000,
                 batch_size=32,
                 epoch_lr_graph=True,
                 tmp_weights_dir='./',
                 visualization_dir='./',
                 visualization_prefix='',
                 verbose=False):
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_lr = num_lr
        self.neighborhood_log_range = neighborhood_log_range
        self.neighborhood_fraction = neighborhood_fraction
        self.epoch_range_factor = epoch_range_factor
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.epoch_lr_graph = epoch_lr_graph
        self.tmp_weights_path = os.path.join(tmp_weights_dir, 'tmp.hdf5')
        self.visualization_dir = visualization_dir
        self.visualization_prefix = visualization_prefix
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.start_lr)

    def on_epoch_begin(self, epoch, logs=None):
        train_gen = TrainDataGenerator(batch_size=self.batch_size,
                                       num_samples=self.num_samples,
                                       rndgray=True)
        val_gen = ValDataGenerator(batch_size=self.batch_size,
                                   rndgray=True)
        self.model.save_weights(self.tmp_weights_path)
        self.losses = []
        history = self.model.history
        current_lr = float(K.get_value(self.model.optimizer.lr))
        neighborhood = int(np.ceil(self.neighborhood_fraction * self.num_lr))
        local_lr_start = current_lr * (10 ** -self.neighborhood_log_range) * (self.epoch_range_factor ** (epoch))
        local_lr_end = current_lr * (10 ** self.neighborhood_log_range) * (self.epoch_range_factor ** (epoch))
        global_geomspace = np.geomspace(self.min_lr, self.max_lr, self.num_lr - neighborhood)
        local_geomspace = np.geomspace(local_lr_start, local_lr_end, neighborhood)
        self.learning_rates = np.sort(np.concatenate([global_geomspace, local_geomspace]))

        if self.verbose:
            print('\n--- Searching among {} learning rates ---'.format(self.num_lr))
        for lr in self.learning_rates:
            history = self.model.fit_generator(generator=train_gen,
                                               validation_data=val_gen,
                                               epochs=1,
                                               use_multiprocessing=True,
                                               workers=12,
                                               verbose=0)
            self.losses.append(history.history['val_loss'][0])
            if self.verbose:
                print('\t> Validation loss for lr={:.3e}: {:.3f}'.format(lr, history.history['val_loss'][0]))
            self.model.load_weights(self.tmp_weights_path)

        best_lr = self.learning_rates[np.argmin(self.losses)]
        K.set_value(self.model.optimizer.lr, best_lr)
        self.model.history = history
        if self.verbose:
            print('\n\tBest learning rate: {:.3e}\n'.format(best_lr))

        if self.epoch_lr_graph:
            plt.figure(figsize=(12, 6))
            plt.plot(self.learning_rates[:len(self.losses)], self.losses, '#800000')
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.xscale('log')
            plt.title('Learning Rate Discovery (Epoch {})'.format(epoch + 1))
            plt.savefig(os.path.join(self.visualization_dir,
                                     '{}_epoch_{:0>3d}.jpg'.format(self.visualization_prefix, epoch + 1)))


class LRDiscovery(Callback):
    """
    Searches for good learning rates at the beginning of training. A learning rate is considered good if it yields
    the best validation accuracy.
    """
    def __init__(self,
                 min_lr,
                 max_lr,
                 num_lr=10,
                 num_samples=10000,
                 batch_size=32,
                 tmp_weights_dir='./',
                 verbose=False):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_lr = num_lr
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.tmp_weights_path = os.path.join(tmp_weights_dir, 'tmp.hdf5')
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        train_gen = TrainDataGenerator(batch_size=self.batch_size,
                                       num_samples=self.num_samples,
                                       rndgray=True)
        val_gen = ValDataGenerator(batch_size=self.batch_size,
                                   rndgray=True)
        self.model.save_weights(self.tmp_weights_path)
        self.losses = []
        history = self.model.history
        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, self.num_lr)

        if self.verbose:
            print('\n--- Searching among {} learning rates ---'.format(self.num_lr))
        for lr in self.learning_rates:
            history = self.model.fit_generator(generator=train_gen,
                                               validation_data=val_gen,
                                               epochs=1,
                                               use_multiprocessing=True,
                                               workers=12,
                                               verbose=0)
            self.losses.append(history.history['val_loss'][0])
            if self.verbose:
                print('\t> Validation loss for lr={:.3e}: {:.3f}'.format(lr, history.history['val_loss'][0]))
            self.model.load_weights(self.tmp_weights_path)

        best_lr = self.learning_rates[np.argmin(self.losses)]
        K.set_value(self.model.optimizer.lr, best_lr)
        self.model.history = history
        if self.verbose:
            print('\n\tBest learning rate: {:.3e}\n'.format(best_lr))
