from base.base_model import BaseModel
import tensorflow as tf
import numpy as np


class MtcnnModel(BaseModel):
    def __init__(self, data_loader, config):
        super(MtcnnModel, self).__init__(config)
        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x = None
        self.y = None
        self.is_training = None
        self.out_argmax = None
        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    def build_model(self):
        """

        :return:
        """

        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        #Convolution Layer #1
        #Computes 32 features using a 3x3 filter with PReLU activation
        #No Padding is added
        #Input Tensor Shape: [batch_size, 48, 48, 3]
        #Output Tensor Shape: [batch_size, 46, 46, 32]
        conv1 = tf.layers.conv2d(
            inputs = self.x,
            filters = 32,
            kernel_size = 3,
            padding = "valid",
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name = "conv1",
            activation = tf.nn.relu)

        #Pooling Layer #1
        #First max pooling layer with a 3x3 filter and a stride of 2
        #Input Tensor Shape: [batch_size, 46, 46, 32]
        #Output Tensor Shape: [batch_size, 23, 23, 32]
        pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 3, strides = 2, name = "pool1")

        #Convolution Layer #2
        #Computes 64 features using a 3x3 filter with PReLU activation
        #Input Tensor Shape: [batch_size, 23, 23, 32]
        #Output Tensor Shape: [batch_size, 21, 21, 64]
        conv2 = tf.layers.conv2d(
            inputs = pool1,
            filters = 64,
            kernel_size = 3,
            padding = "valid",
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name = "conv2",
            activation = tf.nn.relu)

        #Pooling Layer #2
        #Second max pooling layer with a 3x3 filter and a strides of 2
        #Input Tensor Shape: [batch_size, 21, 21, 64]
        #Output Tensor Shape: [batch_size, 10, 10, 64]
        pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 3, strides = 2, name = "pool2")

        #Convolution Layer #3
        #Computes 64 features using a 3x3 filter with PReLU activation
        #Input Tensor Shape: [batch_size, 10, 10, 64]
        #Output Tensor Shape: [batch_size, 8, 8, 64]
        conv3 = tf.layers.conv2d(
            inputs = pool2,
            filters = 64,
            kernel_size = 3,
            padding = "valid",
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name = "conv3",
            activation = tf.nn.relu)

        #Pooling Layer #3
        #Third max pooling layer with a 2x2 filter and a stride of 2
        #Input Tensor Shape: [batch_size, 8, 8, 64]
        #Output Tensor Shape: [batch_size, 4, 4, 64]
        pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = 2, strides = 2, name = "pool3")

        #Convolution Layer #4
        #Computes 128 features using a 2x2 filter with PReLU activation
        #Input Tensor Shape: [batch_size, 4, 4, 64]
        #Output Tensor Shape: [batch_size, 3, 3, 128]
        conv4 = tf.layers.conv2d(
            inputs = pool3,
            filters = 128,
            kernel_size = 2,
            padding = "valid",
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name = "conv4",
            activation = tf.nn.relu)

        #Flatten tensor into a batch of vectors
        #Input Tensor Shape: [batch_size, 3, 3, 128]
        #Output Tensor Shape: [batch_size, 3 * 3 * 128]
        conv4_flat = tf.layers.flatten(inputs = conv4, name = "conv4_flat")

        #Dense Layer
        #Densely connected layer with 256 neurons
        #Input Tensor Shape: [batch_size, 4 * 4 * 128]
        #Output Tensor Shape: [batch_size, 256]
        fc = tf.layers.dense(inputs = conv4_flat, units = 256, name = "fc", activation = tf.nn.relu)

        #Dropout Layer
        #fc = tf.layers.dropout(inputs = fc, rate = 0.25, training = self.is_training, name = "dropout")

        #Logits layer
        #Input Tensor Shape: [batch_size, 256]
        #Output Tensor Shape: [batch_size, 2]
        score = tf.layers.dense(inputs = fc, units = 2, kernel_initializer = tf.initializers.truncated_normal, name = "score")

        prob = tf.nn.softmax(score)

        #Landmark
        #Input Tensor Shape: [batch_size, 256]
        #Output Tensor Shape: [batch_size, 10]
        landmark_pred = tf.layers.dense(inputs = fc, units = 10, kernel_initializer = tf.initializers.truncated_normal, name = "landmark_pred")

        #output
        with tf.variable_scope('outputs'):
            prob_out = tf.identity(prob, "prob_out")
            landmark_out = tf.identity(landmark_pred, "landmark_out")

        label_y = self.y[:,0]
        landmark_y = self.y[:,1:]
        landmark_y = tf.cast(landmark_y, tf.float32)
        loss_prob = tf.losses.sparse_softmax_cross_entropy(labels = label_y, logits = prob)
        loss_landmark_pred = tf.reduce_mean((landmark_y[:,0] - landmark_pred[:,0])**2 +
            (landmark_y[:,1] - landmark_pred[:,1])**2 +
            (landmark_y[:,2] - landmark_pred[:,2])**2 +
            (landmark_y[:,3] - landmark_pred[:,3])**2 +
            (landmark_y[:,4] - landmark_pred[:,4])**2 +
            (landmark_y[:,5] - landmark_pred[:,5])**2 +
            (landmark_y[:,6] - landmark_pred[:,6])**2 +
            (landmark_y[:,7] - landmark_pred[:,7])**2 +
            (landmark_y[:,8] - landmark_pred[:,8])**2 +
            (landmark_y[:,9] - landmark_pred[:,9])**2)

        a = 1.0
        b = 1.0
        self.loss = a * loss_prob + b * loss_landmark_pred

        # accuracy of face
        out_argmax = tf.argmax(score, axis=-1, output_type=tf.int64, name='out_argmax')
        self.acc = tf.reduce_mean(tf.cast(tf.equal(label_y, out_argmax), tf.float32))

        with tf.variable_scope('train_step'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)

    def init_saver(self):
        """
        initialize the tensorflow saver that will be u.is_sed in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)

    @staticmethod
    def conv_bn_relu(name, x, out_filters, kernel_size, training_flag):
        with tf.variable_scope(name):
            out = tf.layers.conv2d(x, out_filters, kernel_size, padding='SAME',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv')
            out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
            out = tf.nn.relu(out)
            return out

    @staticmethod
    def dense_bn_relu_dropout(name, x, num_neurons, dropout_rate, training_flag):
        with tf.variable_scope(name):
            out = tf.layers.dense(x, num_neurons, kernel_initializer=tf.initializers.truncated_normal, name='dense')
            out = tf.layers.batch_normalization(out, training=training_flag, name='bn')
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, dropout_rate, training=training_flag, name='dropout')
            return out
