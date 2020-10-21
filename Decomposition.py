from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.datasets.mnist import load
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p


def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    up_s = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([up_s, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding,
                            strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def build_model():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input(shape=(None, None, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8

    bn = bottleneck(p4, f[4])

    u1_1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2_1 = up_block(u1_1, c3, f[2])  # 16 -> 32
    u3_1 = up_block(u2_1, c2, f[1])  # 32 -> 64
    u4_1 = up_block(u3_1, c1, f[0])  # 64 -> 128

    outputs_1 = keras.layers.Conv2D(
        3, (1, 1), padding="same", activation="sigmoid", name="reflectance_output")(u4_1)

    u1_2 = up_block(bn, c4, f[3])  # 8 -> 16
    u2_2 = up_block(u1_2, c3, f[2])  # 16 -> 32
    u3_2 = up_block(u2_2, c2, f[1])  # 32 -> 64
    u4_2 = up_block(u3_2, c1, f[0])  # 64 -> 128

    outputs_2 = keras.layers.Conv2D(
        3, (1, 1), padding="same", activation="sigmoid", name="shading_output")(u4_2)

    outputs_3 = tf.math.multiply(
        outputs_1, outputs_2, name="reconstruction_output")

    model = keras.models.Model(
        inputs, outputs=(outputs_1, outputs_2, outputs_3))
    return model


model = build_model()
model.summary()

input_high_path = '/content/drive/My Drive/Samsung Project/Decomposition Latest/our485/high'
input_low_path = '/content/drive/My Drive/Samsung Project/Decomposition Latest/our485/low'
#output_reflect_path = '/content/drive/My Drive/Samsung Project/Decomposition_modified/Reflectance'
# can be anything, either the input or the output path
file_path = '/content/drive/My Drive/Samsung Project/Decomposition Latest/our485/high'
# batch_size_loader = 128
# image_crop_y = 128
# image_crop_x = 128

image_y_crop_size = 128
image_x_crop_size = 128
batch_size = 10

##FOR MODEL
input_channels = 3

opt = Adam(lr=0.0001)


##FOR TRAINING
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
#checkpoint_path = 'content/drive/My Drive/Colab Notebooks/Decomposition_modified/Checkpoints/ckpt.hdf5'
current_batch_to_train_on = 0
#epochs = 500
#batch_size = 20
#verbose = 1
save_weights_only = True
lr = 1e-4
momentum = 0.9
decay = 0.0625
epochs = 2000
data_size = 240
# period = 2


#tf.enable_eager_execution()

def random_crop_image_and_labels(input_high, input_low, h, w):
    combined = tf.concat([input_high, input_low], axis=2)
    #combined = tf.concat([combined, reflect_label], axis=2)
    combined_crop = tf.image.random_crop(combined, size=[h, w, 9])
    cropped_input_high = combined_crop[:, :, :3]
    cropped_input_low = combined_crop[:, :, 3:6]
    #cropped_reflect_label = combined_crop[:, :, 6:]
    cropped_input_high.set_shape([h, w, 3])
    cropped_input_low.set_shape([h, w, 3])
    #cropped_shade_label.set_shape([h, w, 3])
    #cropped_reflect_label.set_shape([h, w, 3])
    return cropped_input_high, cropped_input_low


def parse_function(x, y):
    input_high = tf.io.read_file(input_high_path + '/' + x)
    input_high = tf.cast(tf.image.decode_png(
        input_high, channels=3), dtype=tf.float32)
    input_low = tf.io.read_file(input_low_path + '/' + y)
    input_low = tf.cast(tf.image.decode_png(
        input_low, channels=3), dtype=tf.float32)

    input_high, input_low = random_crop_image_and_labels(
        input_high, input_low, image_x_crop_size, image_y_crop_size)

    #img_x, img_y, img_z = random_crop_image_and_labels(img_x, img_y, img_z, image_x_crop_size, image_y_crop_size)

    input_high = input_high/255.0
    input_low = input_low/255.0
    #img_z = img_z/255.0
    # print(img_x.shape, img_y.shape)
    return input_high, input_low
    #return img_x, {"tf_op_layer_reconstruction_output": img_x,"shading_output": img_y,"reflectance_output": img_z}


def parse_function_test(x, y, z):
    img_x = tf.io.read_file(input_path + '/' + x)
    img_x = tf.cast(tf.image.decode_png(img_x, channels=3), dtype=tf.float32)
    img_y = tf.io.read_file(output_shade_path + '/' + y)
    img_y = tf.cast(tf.image.decode_png(img_y, channels=3), dtype=tf.float32)
    img_z = tf.io.read_file(output_reflect_path + '/' + z)
    img_z = tf.cast(tf.image.decode_png(img_z, channels=3), dtype=tf.float32)
    #img_x, img_y, img_z = random_crop_image_and_labels(img_x, img_y, img_z, image_x_crop_size, image_y_crop_size)

    img_x = img_x/255.0
    img_y = img_y/255.0
    img_z = img_z/255.0
    # print(img_x.shape, img_y.shape)
    return img_x


def load_data():
    files = os.listdir(file_path)[:data_size]
    BUFFER_SIZE = len(files)
    ds = tf.data.Dataset.from_tensor_slices((files, files))
    ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.map(parse_function,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    #print(ds)
    return ds


def load_data_test():
    files = os.listdir(file_path)[:data_size]
    BUFFER_SIZE = len(files)
    ds = tf.data.Dataset.from_tensor_slices((files, files, files))
    ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.map(parse_function_test,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)
    #print(ds)
    return ds


def load_images(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32(
        (img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm


train_low_data = []
train_high_data = []
train_low_data_names = glob(input_low_path+'/*.png')
train_low_data_names.sort()
train_high_data_names = glob(input_high_path + '/*.png')
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names)
print('[*] Number of training data: %d' % len(train_low_data_names))
for idx in range(len(train_low_data_names)):
    low_im = load_images(train_low_data_names[idx])
    train_low_data.append(low_im)
    high_im = load_images(train_high_data_names[idx])
    train_high_data.append(high_im)


def gradient(input_tensor, direction):
    smooth_kernel_x = tf.reshape(tf.constant(
        [[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    print(type(kernel))
    print(type(input_tensor))
    gradient_orig = tf.abs(tf.nn.conv2d(
        input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    grad_min = tf.reduce_min(gradient_orig)
    grad_max = tf.reduce_max(gradient_orig)
    grad_norm = tf.div((gradient_orig - grad_min),
                       (grad_max - grad_min + 0.0001))
    return grad_norm


def mutual_i_loss(input_I_low, input_I_high):
    low_gradient_x = gradient(input_I_low, "x")
    high_gradient_x = gradient(input_I_high, "x")
    x_loss = (low_gradient_x + high_gradient_x) * \
        tf.exp(-10*(low_gradient_x+high_gradient_x))
    low_gradient_y = gradient(input_I_low, "y")
    high_gradient_y = gradient(input_I_high, "y")
    y_loss = (low_gradient_y + high_gradient_y) * \
        tf.exp(-10*(low_gradient_y+high_gradient_y))
    mutual_loss = tf.reduce_mean(x_loss + y_loss)
    return mutual_loss


def mutual_i_input_loss(input_I_low, input_im):
    input_gray = tf.image.rgb_to_grayscale(input_im)
    low_gradient_x = gradient(input_I_low, "x")
    input_gradient_x = gradient(input_gray, "x")
    x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
    low_gradient_y = gradient(input_I_low, "y")
    input_gradient_y = gradient(input_gray, "y")
    y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
    mut_loss = tf.reduce_mean(x_loss + y_loss)
    return mut_loss


"""
recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  input_low))
recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - input_high))

equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))

i_mutual_loss = mutual_i_loss(I_low, I_high)

i_input_mutual_loss_high = mutual_i_input_loss(I_high, input_high)
i_input_mutual_loss_low = mutual_i_input_loss(I_low, input_low)

loss_Decom = 1*recon_loss_high + 1*recon_loss_low \
               + 0.01 * equal_R_loss + 0.2*i_mutual_loss \
             + 0.15* i_input_mutual_loss_high + 0.15* i_input_mutual_loss_low
"""

losses = []


def step(input_low, input_high):
    with tf.GradientTape() as tape:
        R_low, I_low, output_low = model(input_low)
        R_high, I_high, output_high = model(input_high)
        loss = 0.01 * tf.reduce_mean(tf.abs(R_low - R_high)) + tf.reduce_mean(tf.abs(output_low - input_low)) + tf.reduce_mean(tf.abs(
            output_high - input_high)) + 0.2 * mutual_i_loss(I_low, I_high) + 0.15*mutual_i_input_loss(I_high, input_high) + 0.15*mutual_i_input_loss(I_low, input_low)
        losses.append(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))


#H = model.fit(train_ds, epochs=1, use_multiprocessing = True, callbacks=[checkpoint, es], verbose=1)
    #pred1 = model(X)
    #loss = categorical_crossentropy(y, pred)
    #grads = tape.gradient(loss, model.trainable_variables)
    #opt.apply_gradients(zip(grads, model.trainable_variables))
print(len(train_low_data))
for data in train_low_data:
  print(type(data))
  print(data.shape)
  break

num_updates = int(len(train_low_data)/batch_size)


def train():
    for epoch in range(epochs):
        for i in range(num_updates):
            start = i*batch_size
            end = start+batch_size
            #.stack(arrays, axis=0)
            input_low = np.stack(train_low_data[start:end], axis=0)
            input_high = np.stack(train_high_data[start:end], axis=0)
            #step(train_high_data[start:end], train_low_data[start:end])
            step(input_high[:, 0:128, 0:128, :], input_low[:, 0:128, 0:128, :])
        print("Epoch : " + str(epoch) + " ....... Done")

        if(epoch % 50 == 0):
          plt.title('Learning Curve')
          plt.xlabel('Epochs')
          plt.ylabel('Loss')
          plt.plot(losses)
          plt.show()


#model.compile(optimizer=opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

#loss, acc = model.evaluate(X_test, y_test)
#print("Test Accuracy : " + str(acc))

#model.save('mnist_model.h5')

train()

input_high_path_eval = '/content/drive/My Drive/Samsung Project/Decomposition Latest/eval15/high'
input_low_path_eval = '/content/drive/My Drive/Samsung Project/Decomposition Latest/eval15/low'

eval_low_data = []
eval_high_data = []
eval_low_data_names = glob(input_low_path_eval+'/*.png')
eval_low_data_names.sort()
eval_high_data_names = glob(input_high_path_eval + '/*.png')
eval_high_data_names.sort()

assert len(eval_low_data_names) == len(eval_high_data_names)

print('[*] Number of eval data: %d' % len(eval_low_data_names))

for idx in range(len(eval_low_data_names)):
    low_im = load_images(eval_low_data_names[idx])
    eval_low_data.append(low_im)
    high_im = load_images(eval_high_data_names[idx])
    eval_high_data.append(high_im)

eval_input_low = np.stack(eval_low_data, axis=0)
eval_input_high = np.stack(eval_high_data, axis=0)

low_eval_output_r, low_eval_output_i, low_eval_output_re = model(
    eval_input_low[:, 0:256, 0:256, :])
high_eval_output_r, high_eval_output_i, high_eval_output_re = model(
    eval_input_high[:, 0:256, 0:256, :])

#output1, output2, output3 = model.predict(test_ds, steps=1)
for i in range(low_eval_output_r.shape[0]):
    img_r = save_img('/content/drive/My Drive/Samsung Project/Decomposition Latest/R_output/R' +
                     str(i)+'.png', low_eval_output_r[i])
    img_s = save_img('/content/drive/My Drive/Samsung Project/Decomposition Latest/S_output/S' +
                     str(i)+'.png', low_eval_output_i[i])
    img_o = save_img('/content/drive/My Drive/Samsung Project/Decomposition Latest/O_output/O' +
                     str(i)+'.png', low_eval_output_re[i])
