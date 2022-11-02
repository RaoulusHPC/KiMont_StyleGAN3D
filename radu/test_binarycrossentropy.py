import tensorflow as tf

bce = tf.losses.BinaryCrossentropy(from_logits=False)
true = tf.zeros(shape=(1, 64))
pred = tf.ones(shape=(1, 64))

print(bce(true,pred))