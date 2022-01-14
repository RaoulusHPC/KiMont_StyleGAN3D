
import datetime
import tensorflow as tf

class TrainingMetrics:

    def __init__(self, *metrics):
        self.metrics = {metric.name:metric for metric in metrics}

    def update(self, name, value):
        self.metrics[name](value)

    def __getitem__(self, name):
        return self.metrics[name].result()

    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset_states()

    def log(self, step):
        if not hasattr(self, 'writer'):
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            self.writer = tf.summary.create_file_writer(train_log_dir)

        with self.writer.as_default():
            for name, metric in self.metrics.items():
                tf.summary.scalar(name, metric.result(), step=step)

    def __repr__(self):
        repr_list = [f"{name: <30s} {metric.result(): 4.5f}" for name, metric in self.metrics.items()]
        return '\n'.join(repr_list)

