import tensorflow as tf
import numpy as np

class ImageLoader:
    """
    DataSetAPI - Load Imgs from the disk
    """

    def __init__(self, config):
        self.config = config

        self.train_imgs_files = []
        self.test_imgs_files = []
        self.train_labels = []
        self.test_labels = []

        data = open(config.file_list, 'r').read().splitlines()
        i = 0
        for d in data:
            if (i < 8000):
                self.train_imgs_files.append(d.split(' ')[0])
                self.train_labels.append([1, int(d.split(' ')[1]), int(d.split(' ')[2]), 
                                             int(d.split(' ')[3]), int(d.split(' ')[4]),
                                             int(d.split(' ')[5]), int(d.split(' ')[6]),
                                             int(d.split(' ')[7]), int(d.split(' ')[8]),
                                             int(d.split(' ')[9]), int(d.split(' ')[10])])
            else:
                self.test_imgs_files.append(d.split(' ')[0])
                self.test_labels.append([1, int(d.split(' ')[1]), int(d.split(' ')[2]),
                                            int(d.split(' ')[3]), int(d.split(' ')[4]),
                                            int(d.split(' ')[5]), int(d.split(' ')[6]),
                                            int(d.split(' ')[7]), int(d.split(' ')[8]),
                                            int(d.split(' ')[9]), int(d.split(' ')[10])])
            i = i + 1

        print("**********************************")
        print(self.train_imgs_files[0])
        print(self.train_labels[0])
        print(self.test_imgs_files[0])
        print(self.test_labels[0])
        print("**********************************")
        self.num_iterations_train = len(self.train_imgs_files) // self.config.batch_size
        self.num_iterations_test = len(self.test_imgs_files) // self.config.batch_size

        self.imgs = tf.convert_to_tensor(self.train_imgs_files, dtype=tf.string)
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.imgs, self.train_labels))
        self.train_dataset = self.train_dataset.map(ImageLoader.parse_train, num_parallel_calls=self.config.batch_size)
        self.train_dataset = self.train_dataset.shuffle(1000, reshuffle_each_iteration=False)
        self.train_dataset = self.train_dataset.batch(self.config.batch_size)

        self.imgs = tf.convert_to_tensor(self.test_imgs_files, dtype=tf.string)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.imgs, self.test_labels))
        self.test_dataset = self.test_dataset.map(ImageLoader.parse_train, num_parallel_calls=self.config.batch_size)
        self.test_dataset = self.test_dataset.shuffle(1000, reshuffle_each_iteration=False)
        self.test_dataset = self.test_dataset.batch(self.config.batch_size)

        self.iterator = tf.data.Iterator.from_structure((tf.float32, tf.int64), ([None, 48, 48, 3], [None,11]))

    @staticmethod
    def parse_train(img_path, label):
        # load img
        img = tf.read_file('../data/image_48x48/' + img_path)
        img = tf.image.decode_png(img, channels=3)
        #img = tf.image.flip_left_right(img)

        return tf.cast(img, tf.float32), tf.cast(label, tf.int64)

    def initialize(self, sess, is_train):
        if is_train:
            sess.run(self.iterator.make_initializer(self.train_dataset))
        else:
            sess.run(self.iterator.make_initializer(self.test_dataset))

    def get_input(self):
        return self.iterator.get_next()

def main():
    class Config:
        file_list = "../data/image_48x48/0list_landmarks_save.txt"

        image_height = 48
        image_width = 48
        batch_size = 64

    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = ImageLoader(Config)

    x, y = data_loader.get_input()

    data_loader.initialize(sess, is_train=True)

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)
    print(out_y)

    data_loader.initialize(sess, is_train=False)

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)
    print(out_y)


if __name__ == '__main__':
    main()
