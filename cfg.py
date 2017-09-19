import random
import os
from PIL import Image
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#图片参数说明

IMAGE_HEIGHT = 40
IMAGE_WIDTH = 132
MAX_CAPTCHA = 4
CHAR_SET_LEN = 62
# train集合存放位置，训练完后预测需注释更改预测图片地址
def get_name_image():
    """
    获取图片和图片名字
    :return:
    """
    all_image = os.listdir('C:/Users/yinghe/pyworkspace/467benz/greypic_train/')
    random_f = random.randint(0, 649)
    base = os.path.basename('C:/Users/yinghe/pyworkspace/467benz/greypic_train/' + all_image[random_f])
    name = os.path.splitext(base)[0]
    image = Image.open('C:/Users/yinghe/pyworkspace/467benz/greypic_train/' + all_image[random_f])
    image = np.array(image)
    return name, image



# get_name_image()
# print(get_name_image())

def name2vec(name):
    """
    名字转向量
    :param name:
    :return:
    """
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
        return k

    for i, c in enumerate(name):
       idx = i * CHAR_SET_LEN + char2pos(c)
       vector[idx] = 1
    return vector

def vec2name(vec):
    """
    向量转名字
    :param vec:
    :return:
    """
    char_pos = vec.nonzero()[0]
    name = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/62
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        else:
            raise ValueError('error')
        name.append(chr(char_code))
    return "".join(name)

# vec = text2vec("F5Sd")
# text = vec2text(vec)
# print(text)  # F5Sd
# vec = text2vec("SFd5")
# text = vec2text(vec)
# print(text)  # SFd5

# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_image()
        batch_x[i, :] = 1*(image.flatten())                         #函数获得的image 是一个含布尔值的矩阵，改行作用是将其转变为只含0，1的40*132的矩阵
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y
####################################################
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    #3 conv
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([5 * 17 * 64, 1024]))    #三层lay，pooling大小为2 生成图片缩小2^3=8倍 除后向上取整
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out

# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)                          #获取预测和实际值对比矩阵如[yes,no,yes,yes,yes,no,no]
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))           #将矩阵转成浮点[1,0,1,1,1,0,0]并计算均值

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于90%,保存模型,完成训练
                if acc > 0.99:
                    saver.save(sess, "./20170914.model", global_step=step)
                    break

            step += 1

train_crack_captcha_cnn()
