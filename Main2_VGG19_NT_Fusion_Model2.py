from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import os
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg19 import VGG19
from keras import optimizers
import pickle
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical

######################### Step1:  Initialize the parameters and file paths #########################
# configure the parameters
batch_size = 32
num_classes = 2
epochs = 5
image_height = 72
image_width = 72

# Set the corresponding file paths
model_folder = 'VGG19_NT_Fusion_Model2_Model'
# Configure the train, val, and test p
base_dir = 'D:/Data/Image/Biomedicine/integrated/MIAS_Patches/MIAS_B_M_Norm_Preprocess/MIAS_NT_ForCNN'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
# test_dir = os.path.join(base_dir, 'test')

# Obtain the data
# Data preprocessing
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    # samplewise_center=True,
    # rescale=1./255,
    rotation_range=180,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen = ImageDataGenerator(
    samplewise_center=True
    # rescale=1./255
)

def extract_features(directory, sample_count,Channels):
    features = np.zeros(shape=(sample_count, image_height, image_width, Channels),dtype=np.float32)
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    i = 0
    for inputs_batch, labels_batch in generator:
        # features_batch = model.predict(inputs_batch)
        features_batch = inputs_batch
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 103447, 3)
train_labels = to_categorical(train_labels)

val_features, val_labels = extract_features(val_dir, 34315, 3)
val_labels = to_categorical(val_labels)

######################### Step2:  Construct the model #########################
base_model = VGG19(weights='imagenet',
                   include_top=False,
                   input_shape=(image_height, image_width, 3))
base_model.summary()
# The index and its layername
# [0, 'input_1']
# [1, 'block1_conv1']
# [2, 'block1_conv2']
# [3, 'block1_pool']
# [4, 'block2_conv1']
# [5, 'block2_conv2']
# [6, 'block2_pool']
# [7, 'block3_conv1']
# [8, 'block3_conv2']
# [9, 'block3_conv3']
# [10, 'block3_conv4']
# [11, 'block3_pool']
# [12, 'block4_conv1']
# [13, 'block4_conv2']
# [14, 'block4_conv3']
# [15, 'block4_conv4']
# [16, 'block4_pool']
# [17, 'block5_conv1']
# [18, 'block5_conv2']
# [19, 'block5_conv3']
# [20, 'block5_conv4']
# [21, 'block5_pool']
layer_conv_base = base_model.layers

for layers_i in range(len(layer_conv_base)):
    print([layers_i,layer_conv_base[layers_i].name])

# configure the input layer
block1_pool_input = layer_conv_base[3].output
block2_pool_input = layer_conv_base[6].output
block3_pool_input = layer_conv_base[11].output
block4_pool_input = layer_conv_base[16].output
block5_pool_input = layer_conv_base[21].output

# add the batch_normalization layer
# For block1_pool
block1_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block1_pool_input)
block1_pool_gap_up = layers.AveragePooling2D(pool_size=(16,16))(block1_pool_norm)
block1_pool_gap_up = layers.Convolution2D(filters = 32, kernel_size=1, activation='relu')(block1_pool_gap_up)
block1_pool_norm_gap = GlobalAveragePooling2D()(block1_pool_gap_up)
# For block2_pool
block2_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block2_pool_input)
block2_pool_gap_up = layers.AveragePooling2D(pool_size=(8,8))(block2_pool_norm)
block2_pool_gap_up = layers.Convolution2D(filters = 32, kernel_size=1, activation='relu')(block2_pool_gap_up)
block2_pool_norm_gap = GlobalAveragePooling2D()(block2_pool_gap_up)
# For block3_pool
block3_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block3_pool_input)
block3_pool_gap_up = layers.AveragePooling2D(pool_size=(4,4))(block3_pool_norm)
block3_pool_gap_up = layers.Convolution2D(filters = 32, kernel_size=1, activation='relu')(block3_pool_gap_up)
block3_pool_norm_gap = GlobalAveragePooling2D()(block3_pool_gap_up)
# For block4_pool
block4_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block4_pool_input)
block4_pool_gap_up = layers.AveragePooling2D(pool_size=(2,2))(block4_pool_norm)
block4_pool_gap_up = layers.Convolution2D(filters = 64, kernel_size=1, activation='relu')(block4_pool_gap_up)
block4_pool_norm_gap = GlobalAveragePooling2D()(block4_pool_gap_up)
# For block5_pool
block5_pool_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(block5_pool_input)
block5_pool_gap_up = layers.Convolution2D(filters = 64, kernel_size=1, activation='relu')(block5_pool_norm)
block5_pool_norm_gap = GlobalAveragePooling2D()(block5_pool_gap_up)

output_fusion_norm = layers.concatenate(
    [block1_pool_norm_gap,block2_pool_norm_gap,block3_pool_norm_gap,block4_pool_norm_gap,block5_pool_norm_gap],
    axis=-1)

# x = Flatten()(x)
# add a global spatial average pooling layer
# x = GlobalAveragePooling2D()(output_fusion_norm)
gap_fusion_norm = layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(output_fusion_norm)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(gap_fusion_norm)
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

layer_conv_base = model.layers
for layers_i in range(len(layer_conv_base)):
    print([layers_i,layer_conv_base[layers_i].name])

for layer in base_model.layers:
    layer.trainable = False

# for layer in model.layers[1:19]:
#     layer.trainable = False

optimizer=optimizers.SGD(lr=0.0001, momentum=0.9)
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              # optimizer=optimizers.RMSprop(lr=1e-4),
              optimizer=optimizer,
              metrics=['accuracy'])

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), model_folder, 'checkpoints')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
filepath_ckp = os.path.join(checkpoint_dir, "weights-improvement-best.hdf5")

# save the best model currently
checkpoint = ModelCheckpoint(
    filepath_ckp,
    monitor='val_loss',
    verbose=2,
    save_best_only=True
    )

# fit setup
print('The traning starts!\n')
# class_weight = {0:5.,
#                 1:10.,
#                 2:1.}

history = model.fit_generator(
                    train_datagen.flow(train_features, train_labels, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(val_features, val_labels),
                    steps_per_epoch = train_features.shape[0] // batch_size,
                    callbacks=[checkpoint],
                    # class_weight = class_weight,
                    verbose=2
                    )

# history = model.fit(train_features, train_labels,
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     validation_split=0.2,
#                     callbacks=[checkpoint],
#                     # class_weight = class_weight,
#                     verbose=2
#                     )

######################### Step3:  Save the history data and plots #########################
# plot the acc and loss figure and save the results
plt_dir = os.path.join(os.getcwd(), model_folder, 'plots')
if not os.path.isdir(plt_dir):
    os.makedirs(plt_dir)

print('The ploting starts!\n')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# save the history
history_dir = os.path.join(os.getcwd(), model_folder, 'history')
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)

# wb 以二进制写入
data_output = open(os.path.join(history_dir,'history_Baseline.pkl'),'wb')
pickle.dump(history.history,data_output)
data_output.close()

# rb 以二进制读取
data_input = open(os.path.join(history_dir,'history_Baseline.pkl'),'rb')
read_data = pickle.load(data_input)
data_input.close()

epochs_range = range(len(acc))
plt.plot(epochs_range, acc, 'ro', label='Training acc')
plt.plot(epochs_range, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(plt_dir, 'acc.jpg'))
plt.figure()

plt.plot(epochs_range, loss, 'ro', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(plt_dir, 'loss.jpg'))
plt.show()
