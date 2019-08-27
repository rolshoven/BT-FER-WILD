from keras_models import RN50, Xception, XceptionCov, InceptionResNetV2, DenseNet121, DenseNet121Cov, DenseNet121CovDropout, STNDenseNet121CovDropout
from resnet50plus import ResNet50Plus
from lightcnn import LightCNN, LightCNNCov
from squeezenet import SqueezeNet
from holonet import HoloNet, HoloNetCov
from residual_attention_network.models import AttentionResNet56
from model import get_model
from ensemble import XceptionResNet50
from generator import TrainDataGenerator, ValDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model
from keras_callback import LRDiscovery
from layers import GlobalCovPooling2D
from datetime import datetime
import os
import argparse

MODELS = ['resnet50', 'resnet50plus', 'xception', 'xceptioncov', 'densenet121', 'densenet121cov', 'densenet121covdropout', 'stndensenet121covdropout', 'inceptionresnetv2',
          'squeezenet', 'attentionresnet56', 'lightcnn', 'lightcnncov', 'holonet', 'holonetcov', 'xceptionresnet50', 'shallowstn']
MODEL_DESCRIPTION = 'Choose one of the folowing models:\n' + \
    ', '.join(MODELS) + '\nDefault: resnet50'

# Argument parser configuration
parser = argparse.ArgumentParser(
    description='Optional arguments for training.')
parser.add_argument('-m', '--model', type=str,
                    help=MODEL_DESCRIPTION, default='resnet50')
parser.add_argument('-s', '--num_samples', type=int,
                    help='Specify the number of samples that are used for training. Default: \'all\'')
parser.add_argument('-e', '--num_epochs', type=int, default=40,
                    help='Specify the number of training epochs. Default: 40')
parser.add_argument('-g', '--gpu', type=int, default=0,
                    help='Specify the GPU for training, either 0 or 1. Default: 0')
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='Specify the batch size. Must be a multiple of 8. Default: 32')
parser.add_argument('--no_multi_processing', dest='multi_processing',
                    action='store_false', help='Deactivate multi-processing.')
parser.set_defaults(multi_processing=True)
parser.add_argument('--initial_epoch', type=int, default=0,
                    help='Specify the starting epoch (important for Tensorboard display). Default: 0')
parser.add_argument('--no_lr_discovery', dest='lr_discovery',
                    action='store_false', help='Deactivate LRDiscovery.')
parser.set_defaults(lr_discovery=True)
parser.add_argument('--multi_gpu', dest='multi_gpu',
                    action='store_true', help='Use both GPUs for training the same model.')
parser.set_defaults(multi_gpu=False)
parser.add_argument('--save_weights_only', dest='save_weights_only',
                    action='store_true', help='Do not save the whole model but just the weights.')
parser.set_defaults(save_weights_only=False)
parser.add_argument('-l', '--load_from_file', dest='load_from_file',
                    action='store_true', help='Load model from file.')
parser.set_defaults(load_from_file=False)
parser.add_argument('--model_path', type=str,
                    help='Path to the model if --load_from_file is used.')
parser.add_argument('--model_name', type=str,
                    help='Name of the model if --load_from_file is used.')
parser.add_argument('--num_workers', type=int, default=12,
                    help='Specify the number of workers for multi-processing. Default: 12')
args = parser.parse_args()
if args.num_samples:
    NUM_SAMPLES = args.num_samples
else:
    NUM_SAMPLES = 'all'

# GPU configuration
if not args.multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Model configuration
if args.load_from_file:
    if not args.model_path or not args.model_name:
        raise Exception(
            'When using --load_from_file, arguments --model_path and --model_name must be specified.')
    custom_objects = {'GlobalCovPooling2D': GlobalCovPooling2D}
    MODEL_NAME = args.model_name
    model = load_model(args.model_path, custom_objects=custom_objects)
else:
    if args.model == 'resnet50':
        MODEL_NAME = 'ResNet50'
        model = RN50()
    elif args.model == 'resnet50plus':
        MODEL_NAME = 'ResNet50Plus'
        model = ResNet50Plus()
    elif args.model == 'xception':
        MODEL_NAME = 'Xception'
        model = Xception()
    elif args.model == 'xceptioncov':
        MODEL_NAME = 'XceptionCov'
        model = XceptionCov()
    elif args.model == 'densenet121':
        MODEL_NAME = 'DenseNet121'
        model = DenseNet121()
    elif args.model == 'densenet121cov':
        MODEL_NAME = 'DenseNet121Cov'
        model = DenseNet121Cov()
    elif args.model == 'densenet121covdropout':
        MODEL_NAME = 'DenseNet121CovDropout'
        model = DenseNet121CovDropout()
    elif args.model == 'stndensenet121covdropout':
        MODEL_NAME = 'STNDenseNet121CovDropout'
        model = STNDenseNet121CovDropout()
    elif args.model == 'inceptionresnetv2':
        MODEL_NAME = 'InceptionResNetV2'
        model = InceptionResNetV2()
    elif args.model == 'squeezenet':
        MODEL_NAME = 'SqueezeNet'
        model = SqueezeNet()
    elif args.model == 'attentionresnet56':
        MODEL_NAME = 'AttentionResNet56'
        model = AttentionResNet56(
            shape=(192, 192, 3), n_classes=8, n_channels=32)
    elif args.model == 'lightcnn':
        MODEL_NAME = 'LightCNN'
        model = LightCNN()
    elif args.model == 'lightcnncov':
        MODEL_NAME = 'LightCNNCov'
        model = LightCNNCov()
    elif args.model == 'holonet':
        MODEL_NAME = 'HoloNet'
        model = HoloNet()
    elif args.model == 'holonetcov':
        MODEL_NAME = 'HoloNetCov'
        model = HoloNetCov()
    elif args.model == 'xceptionresnet50':
        MODEL_NAME = 'XceptionResNet50'
        model = XceptionResNet50()
    elif args.model == 'shallowstn':
        MODEL_NAME = 'ShallowSTN'
        model = get_model()
    else:
        raise Exception(
            '\'{}\' is no valid argument for the model specification.'.format(args.model))

if args.multi_gpu:
    model = multi_gpu_model(model, gpus=2)

model.compile(optimizer=Adam(lr=2.512e-04, clipnorm=1.0),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model output path configuration
BASE_PATH = '/data/cvg/luca/FER/models'
path_csvlogger = os.path.join(BASE_PATH, MODEL_NAME, 'logs', 'csv_logger')
path_tensorboard = os.path.join(BASE_PATH, MODEL_NAME, 'logs', 'tensorboard')
path_model_save = os.path.join(BASE_PATH, MODEL_NAME, 'keras_models')
path_lrdiscovery = os.path.join(BASE_PATH, MODEL_NAME, 'lr_discovery')
os.makedirs(path_csvlogger, exist_ok=True)
os.makedirs(path_tensorboard, exist_ok=True)
os.makedirs(path_model_save, exist_ok=True)
os.makedirs(path_lrdiscovery, exist_ok=True)

# Keras callback configuration
start_datetime = datetime.now().strftime('%m.%d-%H.%M')
lr_discovery = LRDiscovery(min_lr=1e-7, max_lr=1e-1, num_lr=10, num_samples=10000, batch_size=args.batch_size,
                           verbose=True, tmp_weights_dir='/data/cvg/luca/temp')
tensorboard = TensorBoard(log_dir=path_tensorboard,
                          histogram_freq=0, write_graph=True, write_images=True)
if args.save_weights_only:
    checkpoint = ModelCheckpoint(os.path.join(path_model_save, start_datetime + '-WEIGHTS-E-{epoch:02d}-VA-{val_acc:.2f}.hdf5'),
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 verbose=0)
else:
    checkpoint = ModelCheckpoint(os.path.join(path_model_save, start_datetime + '-MODEL-E-{epoch:02d}-VA-{val_acc:.2f}.hdf5'),
                                 monitor='val_loss',
                                 verbose=0)
logger = CSVLogger(os.path.join(path_csvlogger, '{}.csv'.format(
    start_datetime)), separator=',', append=False)

callbacks = [tensorboard, checkpoint, logger]

if args.lr_discovery:
    callbacks.append(lr_discovery)

# def vggface_preprocessing(img):
#     return preprocess_input(img, data_format='channels_last', version=2)

# When using VGGFace for transfer learning, make sure to pass vggface_preprocessing as preprocess_func argument
# and set range255=True. Moreover, the image size needs to be at least 197x197, you might want to use pad_to_size=197.

# Data generator configuration
train_data_generator = TrainDataGenerator(batch_size=args.batch_size,
                                          num_samples=NUM_SAMPLES,
                                          rndgray=True)

val_data_generator = ValDataGenerator(batch_size=args.batch_size,
                                      augment=False)

# Training configuration
history = model.fit_generator(generator=train_data_generator,
                              validation_data=val_data_generator,
                              epochs=args.num_epochs,
                              use_multiprocessing=args.multi_processing,
                              workers=args.num_workers,
                              initial_epoch=args.initial_epoch,
                              verbose=1,
                              callbacks=callbacks)
