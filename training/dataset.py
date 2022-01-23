
import tensorflow as tf


label_map = {
    'Articulations, eyelets and other articulated joints': 0,
    'Bearing accessories': 1,
    'Bushes': 2,
    'Cap nuts': 3,
    'Castle nuts': 4,
    'Castor': 5,
    'Chain drives': 6,
    'Clamps': 7,
    'Collars': 8,
    'Conventional rivets': 9,
    'Convex washer': 10,
    'Cylindrical pins': 11,
    'Elbow fitting': 12,
    'Eye screws': 13,
    'Fan': 14,
    'Flange nut': 15,
    'Flanged block bearing': 16,
    'Flanged plain bearings': 17,
    'Grooved pins': 18,
    'Helical geared motors': 19,
    'Hexagonal nuts': 20,
    'Hinge': 21,
    'Hook': 22,
    'Impeller': 23,
    'Keys and keyways, splines': 24,
    'Knob': 25,
    'Lever': 26,
    'Locating pins': 27,
    'Locknuts': 28,
    'Lockwashers': 29,
    'Nozzle': 30,
    'Plain guidings': 31,
    'Plates, circulate plates': 32,
    'Plugs': 33,
    'Pulleys': 34,
    'Radial contact ball bearings': 35,
    'Right angular gearings': 36,
    'Right spur gears': 37,
    'Rivet nut': 38,
    'Roll pins': 39,
    'Screws and bolts with countersunk head': 40,
    'Screws and bolts with cylindrical head': 41,
    'Screws and bolts with hexagonal head': 42,
    'Setscrew': 43,
    'Slotted nuts': 44,
    'Snap rings': 45,
    'Socket': 46,
    'Spacers': 47,
    'Split pins': 48,
    'Spring washers': 49,
    'Springs': 50,
    'Square': 51,
    'Square nuts': 52,
    'Standard fitting': 53,
    'Studs': 54,
    'Switch': 55,
    'T-nut': 56,
    'T-shape fitting': 57,
    'Taper pins': 58,
    'Tapping screws': 59,
    'Threaded rods': 60,
    'Thrust washers': 61,
    'Toothed': 62,
    'Turbine': 63,
    'Valve': 64,
    'Washer bolt': 65,
    'Wheel': 66,
    'Wingnuts': 67
}

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
        bias = drange_out[0] - drange_in[0] * scale
        data = data * scale + bias
    return data

def get_random_labels(batch_size, label_size):
    if not label_size:
        return None
    labels = tf.random.uniform(shape=(batch_size,), minval=0, maxval=label_size, dtype='int32')
    labels = tf.one_hot(indices=labels, depth=label_size)
    return labels

def augment(x):
    # flip = tf.random.uniform([3])
    # if flip[0] < 0.5:
    #     x = tf.reverse(x, axis=[0]) 
    # if flip[1] < 0.5:
    #     x = tf.reverse(x, axis=[1]) 
    # if flip[2] < 0.5:
    #     x = tf.reverse(x, axis=[2])

    # for axes in [(0, 1), (1, 2), (0, 2)]:
    #     if tf.random.uniform([]) < 0.5:
    #         if tf.random.uniform([]) < 0.5:
    #             x = tf.experimental.numpy.rot90(x, k=1, axes=axes) 
    #         else:
    #             x = tf.experimental.numpy.rot90(x, k=3, axes=axes) 
    return x

def simple_grab_aug(components, label):
    comp1, comp2 = components
    comp1 = augment(comp1)
    comp2 = augment(comp2)
    if tf.random.uniform([]) < 0.5:
        comp1, comp2 = comp2, comp1
        label = tf.cond(label > 0.5, lambda : 0., lambda : 1.)
    return (comp1, comp2), label

def get_abc_dataset(tfrecords, batch_size: int=1, repeat: int=1, augment_function: callable=None):
    raw_component_dataset = tf.data.TFRecordDataset(tfrecords, compression_type='GZIP')

    # Create a dictionary describing the features.
    component_feature_description = {
        'component_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def read_tfrecord(serialized_example):
        # Parse the input tf.train.Example proto using the dictionary above.
        example = tf.io.parse_single_example(serialized_example, component_feature_description)
        component = tf.io.parse_tensor(example['component_raw'], out_type=bool)
        component = tf.cast(component, 'float32')
        component = 2 * component - 1
        component = component[..., tf.newaxis]
        return component, tf.zeros(shape=())

    dataset = raw_component_dataset.map(read_tfrecord)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_labeled_dataset(tfrecords, batch_size: int=1, repeat: int=1, augment_function: callable=None):
    raw_component_dataset = tf.data.TFRecordDataset(tfrecords, compression_type='GZIP')

    # Create a dictionary describing the features.
    component_feature_description = {
        'component_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def read_tfrecord(serialized_example):
        # Parse the input tf.train.Example proto using the dictionary above.
        example = tf.io.parse_single_example(serialized_example, component_feature_description)
        component = tf.io.parse_tensor(example['component_raw'], out_type=bool)
        label = tf.io.parse_tensor(example['label_raw'], out_type=float)
        component = tf.cast(component, 'float32')
        component = tf.expand_dims(component, axis=-1)
        return component, label

    dataset = raw_component_dataset.map(read_tfrecord)
    if augment_function:
        dataset = dataset.map(augment_function)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_mcb_dataset(tfrecords, batch_size: int=1, repeat: int=1, augment_function: callable=None):
    raw_component_dataset = tf.data.TFRecordDataset(tfrecords, compression_type='GZIP')

    # Create a dictionary describing the features.
    component_feature_description = {
        'component_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def read_tfrecord(serialized_example):
        # Parse the input tf.train.Example proto using the dictionary above.
        example = tf.io.parse_single_example(serialized_example, component_feature_description)
        component = tf.io.parse_tensor(example['component_raw'], out_type='bool')
        component = tf.cast(component, 'float32')
        component = 2 * component - 1
        component = component[..., tf.newaxis]
        label = tf.io.parse_tensor(example['label_raw'], out_type='float32')
        return component, label

    dataset = raw_component_dataset.map(read_tfrecord)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.repeat(repeat)
    if augment_function:
        dataset = dataset.map(lambda x, y: (augment_function(x), y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_simplegrab_dataset(tfrecords):
    raw_component_dataset = tf.data.TFRecordDataset(tfrecords, compression_type='GZIP')

    # Create a dictionary describing the features.
    component_feature_description = {
        'component1_raw': tf.io.FixedLenFeature([], tf.string),
        'component2_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }

    def read_tfrecord(serialized_example):
        # Parse the input tf.train.Example proto using the dictionary above.
        example = tf.io.parse_single_example(serialized_example, component_feature_description)
        component1 = tf.io.parse_tensor(example['component1_raw'], out_type='bool')
        component1 = tf.cast(component1, 'float32')
        component1 = component1[..., tf.newaxis]
        component1 = 2 * component1 - 1
        component2 = tf.io.parse_tensor(example['component2_raw'], out_type='bool')
        component2 = tf.cast(component2, 'float32')
        component2 = component2[..., tf.newaxis]
        component2 = 2 * component2 - 1
        label = example['label']
        return (component1, component2), label
        
    dataset = raw_component_dataset.map(read_tfrecord)
    return dataset



def get_projected_dataset(tfrecords):
    raw_component_dataset = tf.data.TFRecordDataset(tfrecords, compression_type='GZIP')
    # Create a dictionary describing the features.
    component_feature_description = {
        'original_image_raw': tf.io.FixedLenFeature([], tf.string),
        'generated_image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
        'w_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def read_tfrecord(serialized_example):
        # Parse the input tf.train.Example proto using the dictionary above.
        example = tf.io.parse_single_example(serialized_example, component_feature_description)
        original_image = tf.io.parse_tensor(example['original_image_raw'], out_type='float32')[0]
        generated_image = tf.io.parse_tensor(example['generated_image_raw'], out_type='float32')[0]
        label = tf.io.parse_tensor(example['label_raw'], out_type='float32')[0]
        w = tf.io.parse_tensor(example['w_raw'], out_type='float32')[0]

        return original_image, generated_image, label, w

    dataset = raw_component_dataset.map(read_tfrecord)
    return dataset

if __name__ == "__main__":
    tfrecords = []
    train_dataset = get_mcb_dataset(tfrecords, label_size=9, batch_size=16, repeat=1, augment_function=None)
    dist_dataset = train_dataset
    next_label = 0
    from training.visualize import screenshot_and_save
    for data, label in dist_dataset:
        if tf.math.reduce_mean(label[:, next_label]) == 1:
            screenshot_and_save(list(data), f'results/mcb32_balanced/{next_label}.png')
            next_label += 1
        if next_label == 9:
            break