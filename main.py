import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
#            'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emotion = {'Angry': 0, 'Fear': 1, 'Happy': 2,
           'Sad': 3, 'Surprise': 4, 'Neutral': 5}
emo = ['Angry', 'Fear', 'Happy',
       'Sad', 'Surprise', 'Neutral']

def main():
    read_data()

def reconstruct(pix_str, size=(48, 48)):
    pix_arr = np.array([int(n) for n in pix_str.split()])
    # print(temp_str)
    # return pix_arr.reshape(size)
    return pix_arr

def label_fix(label):
    if label == 1 or label == 0:
        return 0
    elif label == 2:
        return 1
    elif label == 3:
        return 2
    elif label == 4:
        return 3
    elif label == 5:
        return 4
    elif label == 6:
        return 5

def prepare(train_data, test_data):
    scaler = MinMaxScaler()
    print(type(train_data))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.fit_transform(test_data)

    return train_scaled, test_scaled

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0,1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    # x --> [batch, H, w, Channels]
    # w --> [filter H, filter W, Channels IN, Channels OUT]

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# CONVOLUTIONAL LAYER
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)

# NORMAL (FULLY CONNECTED)
def normal_full_layer(input_layer, size):
    print("input layer shape")
    print(input_layer.get_shape()[1])
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def read_data():
    # print data.tail()

    data = pd.read_csv("./data/fer2013.csv")
    print(data.Usage.value_counts())

    data["emotion"] = data["emotion"].apply(label_fix)
    train_data = data[data.Usage == "Training"]
    test_data = data[data.Usage == "PrivateTest"]


    train_emotion_one_hot = np.zeros((len(train_data), 6))
    test_emotion_one_hot = np.zeros((len(test_data), 6))

    for i, index in enumerate(train_data["emotion"]):
        train_emotion_one_hot[i, index] = 1

    for i, index in enumerate(test_data["emotion"]):
        test_emotion_one_hot[i, index] = 1

    # remove all columns that are not matrix data
    train_data = train_data.drop(["Usage","emotion"], axis=1)
    test_data = test_data.drop(["Usage", "emotion"], axis=1)

    train_data['pixels'] = train_data.pixels.apply(lambda x: reconstruct(x))
    test_data['pixels'] = test_data.pixels.apply(lambda x: reconstruct(x))

    train_pixels = np.array(train_data['pixels'].values.tolist())
    test_pixels = np.array(test_data['pixels'].values.tolist())

    train_data, test_data = prepare(train_pixels, test_pixels)

    print("this is after prepared")
    print(train_data)

    print(train_data)
    training(train_data, test_data, train_emotion_one_hot, test_emotion_one_hot)


def predict():
    print("this is the prediction")


def training(train_data, test_data, train_emotion_one_hot, test_emotion_one_hot):
    x = tf.placeholder(tf.float32, shape=[None, 2304])
    y_true = tf.placeholder(tf.float32, shape=[None, 6])

    # layers
    x_image = tf.reshape(x, [-1, 48, 48, 1])

    convo_1 = convolutional_layer(x_image, shape=[3, 3, 1, 32])
    convo_1_pooling = max_pool_2by2(convo_1)

    convo_2 = convolutional_layer(convo_1_pooling, shape=[3, 3, 32, 64])
    convo_2_pooling = max_pool_2by2(convo_2)

    convo_3 = convolutional_layer(convo_2_pooling, shape=[3, 3, 64, 128])
    convo_3_pooling = max_pool_2by2(convo_3)

    convo_4 = convolutional_layer(convo_3_pooling, shape=[3, 3, 128, 256])
    convo_4_pooling = max_pool_2by2(convo_4)

    convo_4_flat = tf.reshape(convo_4_pooling, [-1, 3*3*256])
    full_layer_1 = normal_full_layer(convo_4_flat, 4096)
    full_layer_2 = tf.nn.relu(normal_full_layer(full_layer_1, 1024))


    hold_prob = tf.placeholder(tf.float32)
    full_one_dropout = tf.nn.dropout(full_layer_2, keep_prob=hold_prob)

    y_pred = normal_full_layer(full_one_dropout, 6)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_entropy)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    steps = 100000
    # steps = 5000

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)

        for i in range(steps):
            batch_x, batch_y = next_batch(50, train_data, train_emotion_one_hot)

            sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

            if i%100 == 0:
                print("ON STEP: {}".format(i))
                print("ACCURACY: ")

                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true,1))
                acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                print(sess.run(acc, feed_dict={x: test_data, y_true: test_emotion_one_hot, hold_prob: 1.0}))
                print('\n')

        saver.save(sess, "./model2/emotion_model_final")


if __name__ == "__main__":
    main()