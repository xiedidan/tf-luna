import argparse
import sys

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

Args = None

def resnet(input):
    pass

def main(_):
    dataset = dataLoader.load(Args.data_dir)

    # input = slice + slice location [x, y, z]
    input = tf.placeholder(tf.float32, [None, Args.side_length ** 3 + 3])
    label = tf.placeholder(tf.float32, [None, Args.side_length ** 3])
    output = resnet(input)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)

    with tf.name_scope('optimizer'):
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cross_entropy)

    # with tf.name_scope('accuracy'):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(Args.epoch):
            batch = dataset.load_train(Args.batch_size)
            
            if i % Args.evaluation_interval == 0:
                print('Step {0}, training loss {1}'.format(i, cross_entropy))
            trainer.run(feed_dict={input: batch[0], label: batch[1]})

            if i % Args.test_interval == 0:
                test_batch = dataset.load_test(Args.test_step)
                print('Step {0}, test loss {1}'.format(i, cross_entropy.eval(feed_dict={input: test_batch[0], label: test_batch[1]})))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data', help='Directory for data sets')
    parser.add_argument('--side_length', type=int, default=64, help='Slice side length')
    parser.add_argument('--epoch', type=int, default=10000, help='Max training epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size of each epoch')
    parser.add_argument('--evaluation_interval', type=int, default=100, help='Epoch between each evaluation output')
    parser.add_argument('--test_interval', type=int, default=1000, help='Epoch between each test')
    parser.add_argument('--test_step', type=int, default=25, help='Epoch of each test')
    Args, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
