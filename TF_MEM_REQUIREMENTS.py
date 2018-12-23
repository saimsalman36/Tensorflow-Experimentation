import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.sparse import coo_matrix

from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer

import sys

K = int(sys.argv[1])
DIMENSION = int(K ** 2 / 2)
BATCH_SIZE = 10
K_ACTIONS_PERCENTAGE = 20.0

A_SIZE = int(DIMENSION * (DIMENSION - 1) / 2)
K_ACTIONS = int(K_ACTIONS_PERCENTAGE/100.0*(A_SIZE*1.0))

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float16)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

inputs_tm = tf.placeholder(
                shape= [None, DIMENSION, DIMENSION, 2],
                dtype=tf.float16,
                name='in_tm')

conv_layer_1 = slim.conv2d(activation_fn=tf.nn.elu, 
                inputs=inputs_tm, num_outputs=16,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_size=[4, 4],
                stride =[1, 1],
                padding='VALID', scope=('conv-1'))

conv_layer_2 = slim.conv2d(activation_fn=tf.nn.elu, 
                inputs=conv_layer_1, num_outputs=32,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_size=[4, 4],
                stride =[1, 1],
                padding='VALID', scope=('conv-2'))

hidden = slim.fully_connected(slim.flatten(conv_layer_2), 256, activation_fn=tf.nn.elu, scope='fc0') # [?, 256]

policy = tf.contrib.layers.fully_connected(hidden,A_SIZE,
        activation_fn=tf.nn.softmax,
        weights_initializer=normalized_columns_initializer(0.01),
        biases_initializer=None) # [?, a_size] -> [?, 28]

value = tf.contrib.layers.fully_connected (hidden,K_ACTIONS,
        activation_fn=None,
        weights_initializer=normalized_columns_initializer(1.0),
        biases_initializer=None) # [?, k_acts] -> [?, 6]

policy_sum = tf.math.reduce_sum(policy)
value_sum = tf.math.reduce_sum(value)

sum_ = policy_sum + value_sum


run_metadata = tf.RunMetadata()
sess = tf.Session()
sess.run(tf.global_variables_initializer(),
        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=run_metadata)

for i in xrange(100):
    print sess.run(sum_, feed_dict = {inputs_tm: np.random.rand(BATCH_SIZE, DIMENSION,DIMENSION, 2)})

print "\n\n *** PARAMETERS AND SHAPES *** \n\n"
# Print trainable variable parameter statistics to stdout.
ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder

param_stats = tf.profiler.profile(
    tf.get_default_graph(),
    options=ProfileOptionBuilder.trainable_variables_parameter())

# Use code view to associate statistics with Python codes.
opts = ProfileOptionBuilder(
    ProfileOptionBuilder.trainable_variables_parameter()
    ).with_node_names(show_name_regexes=['.*my_code1.py.*', '.*my_code2.py.*']
    ).build()
param_stats = tf.profiler.profile(
    tf.get_default_graph(),
    cmd='code',
    options=opts)

print('total_params: %d\n' % param_stats.total_parameters)

print "\n\n *** FLOATING POINT OPERATIONS *** \n\n"
tf.profiler.profile( tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())

print "\n\n *** MEMORY + TIMING INFORMATION *** \n\n"
# Print to stdout an analysis of the memory usage and the timing information
# broken down by python codes.
ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
    ).with_node_names(show_name_regexes=['.*my_code.py.*']).build()

tf.profiler.profile(
    tf.get_default_graph(),
    run_meta=run_metadata,
    cmd='code',
    options=opts)

# Print to stdout an analysis of the memory usage and the timing information
# broken down by operation types.
tf.profiler.profile(
    tf.get_default_graph(),
    run_meta=run_metadata,
    cmd='op',
    options=tf.profiler.ProfileOptionBuilder.time_and_memory())

fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open('timeline_01.json', 'w') as f:
    f.write(chrome_trace)