# tensorflow 2.0
#import tensorflow as tf
#import os


# Specify which GPU(s) to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

# On CPU/GPU placement
#config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#config.gpu_options.allow_growth = True
#tf.compat.v1.Session(config=config)

# tensorflow 1.14
import tensorflow as tf
import numpy as np
import time

config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
config.gpu_options.visible_device_list="0"

input = tf.placeholder(tf.float32, [1000])
data = np.random.rand(1000).astype('float32')
output = tf.nn.softmax(tf.nn.relu(input))

with tf.Session(config=config) as sess:
      res = sess.run(output, feed_dict={input: data})
      print("Session executed, check devices with nvidia-smi")
      time.sleep(20)
      print("Exiting")
