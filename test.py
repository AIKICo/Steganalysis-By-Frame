import tensorflow as tf
import keras
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
b'Hello, TensorFlow!'