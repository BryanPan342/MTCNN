import argparse
import os
import sys
import cv2
from typing import Iterable
import tensorflow as tf


def LoadModel(modelFile):
    try:
        with tf.gfile.GFile(modelFile, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
    except BaseException as e:
        print("Fail to load model file %s" % (modelFile))
        return None

    try:
        assert graph_def is not None
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name='',
                producer_op_list=None
            )
    except BaseException as e:
        print("Fail to import graph")
        return None

    return graph


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('modelFile', type=str, help='The file name of the frozen graph.')
    parser.add_argument('imagePath', type=str, help='The path of input images.')
    args = parser.parse_args()

    if not os.path.exists(args.modelFile):
        parser.exit(1, 'The specified file does not exist: {}'.format(args.modelFile))

    if not os.path.exists(args.imagePath):
        parser.exit(1, 'The specified file does not exist: {}'.format(args.imagePath))

    # load model
    graph = LoadModel(args.modelFile)
    assert graph is not None

    # model input and output tensor
    image_in = graph.get_tensor_by_name("inputs/IteratorGetNext:0")
    prob_out = graph.get_tensor_by_name("outputs/prob_out:0")
    landmark_out = graph.get_tensor_by_name("outputs/landmark_out:0")
    height = image_in.shape[1]
    width  = image_in.shape[2]
    chan   = image_in.shape[3]
    print("input image dimension %dx%dx%d" % (height, width, chan))

    # loop through all the images in image path
    for imageFile in os.listdir(args.imagePath):

        # if they are *.bmp/jpg/png
        if imageFile.endswith(".bmp") or imageFile.endswith('.jpg') or imageFile.endswith('.png'   ):

            # read image by opencv, and resize, change order to RGB, change format to float
            img = cv2.imread(os.path.join(args.imagePath, imageFile))
            if img is None:
                print('Cannot open file: %s' % (imageFile))
                continue
            image = cv2.resize(img, (width, height), cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(float)

            # evaluate prob_out and landmark_out by tensorflow graph
            with tf.Session(graph = graph) as sess:
                prob_results = prob_out.eval({image_in: [image]})
                landmark_results = landmark_out.eval({image_in: [image]})

            # print results of prob_out and landmark_out
            print('prob_out: ')
            print(prob_results)
            print('landmark_out: ')
            print(landmark_results)


            # Resize
            scale = 3
            img = cv2.resize(img, (width * scale, height * scale), cv2.INTER_CUBIC)

            # show landmarks on image
            ipx0 = int(landmark_results[0,0] * scale + .5)
            ipy0 = int(landmark_results[0,1] * scale + .5)
            ipx1 = int(landmark_results[0,2] * scale + .5)
            ipy1 = int(landmark_results[0,3] * scale + .5)
            ipx2 = int(landmark_results[0,4] * scale + .5)
            ipy2 = int(landmark_results[0,5] * scale + .5)
            ipx3 = int(landmark_results[0,6] * scale + .5)
            ipy3 = int(landmark_results[0,7] * scale + .5)
            ipx4 = int(landmark_results[0,8] * scale + .5)
            ipy4 = int(landmark_results[0,9] * scale + .5)

            cv2.rectangle(img, (ipx0-1, ipy0-1), (ipx0+1, ipy0+1), (0, 255, 0), 1)
            cv2.rectangle(img, (ipx1-1, ipy1-1), (ipx1+1, ipy1+1), (0, 255, 0), 1)
            cv2.rectangle(img, (ipx2-1, ipy2-1), (ipx2+1, ipy2+1), (0, 255, 0), 1)
            cv2.rectangle(img, (ipx3-1, ipy3-1), (ipx3+1, ipy3+1), (0, 255, 0), 1)
            cv2.rectangle(img, (ipx4-1, ipy4-1), (ipx4+1, ipy4+1), (0, 255, 0), 1)

            # show image by opencv
            cv2.imshow('image', img)

            # wait for key by opencv, it key is Escape, break
            k = cv2.waitKey(0)
            if k == 27:
                break
            


# entry
if __name__ == "__main__":
    print(tf.__version__)
    main()
