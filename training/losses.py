import tensorflow as tf

def generator_logistic(fake):
    loss = -tf.nn.softplus(-fake)
    return loss
    
def generator_logistic_ns(fake):
    loss = tf.nn.softplus(-fake)
    return loss

# 1 ist richtiges
#0 ist falsches

def discriminator_logistic(real, fake):
    loss = tf.nn.softplus(fake)
    loss += tf.nn.softplus(-real)
    return loss

def generator_wgan(fake):
    loss = -fake
    return loss

def discriminator_wgan(real, fake):
    loss = fake - real
    loss += tf.math.square(real) * 0.001
    return loss
    
# TODO: ausdenken was Sinn macht
def label_loss(predicted_label, target_label):
    return tf.math.square(predicted_label - target_label)