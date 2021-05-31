import tensorflow as tf
import tensorflow.contrib as tf_contrib


weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
"""
pad = (k-1) // 2
size = (I-k+1+2p) // s
"""
def conv(x, channels, kernel=4, stride=2, pad=1,  scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init, strides=stride)

        return x


def deconv(x, channels, kernel=4, stride=2, scope='deconv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init, strides=stride, padding='SAME')

        return x


def resblock(x_init, channels,  scope='resblock_0'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = tf.pad(x_init, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=3, kernel_initializer=weight_init, strides=1)
            x = batch_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=3, kernel_initializer=weight_init, strides=1)
            x = batch_norm(x)

        return x + x_init

    # def batch_norm(x, training, name):
    #     training = tf.cast(training, tf.bool)
    #     with tf.variable_scope(name):
    #         x = tf.cond(training,
    #                     lambda: slim.batch_norm(x, decay=0.997, epsilon=1e-5, scale=True, is_training=training,
    #                                             scope=name + '_batch_norm', reuse=None),
    #                     lambda: slim.batch_norm(x, decay=0.997, epsilon=1e-5, scale=True, is_training=training,
    #                                             scope=name + '_batch_norm', reuse=True))
    #     return x
    #
    # def conv_layer(x, training, filters, name):
    #     batch_norm_params = {
    #         'decay': 0.997,
    #         'epsilon': 1e-5,
    #         'scale': True,
    #         'is_training': training
    #     }
    #     with tf.variable_scope(name):
    #         with slim.arg_scope([slim.conv2d],
    #                             activation_fn=tf.nn.relu,
    #                             normalizer_fn=slim.batch_norm,
    #                             normalizer_params=batch_norm_params,
    #                             weights_regularizer=slim.l2_regularizer(1e-5)):
    #             x = slim.conv2d(x, filters, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu,
    #                             scope=name + '_conv3x3')
    #             # x = slim.dropout(x, keep_prob=0.2, is_training=training, scope=name+'_dropout')
    #     return x
    #
    # def dense_block(x, training, block_nb, name):
    #     dense_out = []
    #     with tf.variable_scope(name):
    #         for i in range(layers_per_block[block_nb]):
    #             conv = conv_layer(x, training, growth_k, name=name + '_layer_' + str(i))
    #             x = tf.concat([conv, x], axis=3)
    #             dense_out.append(conv)
    #         x = tf.concat(dense_out, axis=3)
    #     return x
    #
    # def transition_down(x, training, filters, name):
    #     batch_norm_params = {
    #         'decay': 0.997,
    #         'epsilon': 1e-5,
    #         'scale': True,
    #         'is_training': training
    #     }
    #     with tf.variable_scope(name):
    #         x = slim.conv2d(x, filters, kernel_size=1, stride=1, padding='SAME', normalizer_fn=slim.batch_norm,
    #                         normalizer_params=batch_norm_params,
    #                         weights_regularizer=slim.l2_regularizer(1e-5), activation_fn=tf.nn.relu,
    #                         scope=name + '_conv1x1')
    #         x = slim.dropout(x, keep_prob=0.2, is_training=training, scope=name + '_dropout')
    #         x = slim.max_pool2d(x, kernel_size=4, stride=2, padding='SAME', scope=name + '_maxpool2x2')
    #     return x
    #
    # def transition_up(x, filters, name):
    #     with tf.variable_scope(name):
    #         x = slim.conv2d_transpose(x, filters, kernel_size=3, stride=2,
    #                                   padding='SAME', activation_fn=tf.nn.relu, scope=name + '_trans_conv3x3')
    #     return x
    #
def flatten(x) :
    return tf.layers.flatten(x)

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True, updates_collections=None, is_training=is_training, scope=scope)


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss



def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss
