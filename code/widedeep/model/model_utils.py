from widedeep.utils import get_logger, VariableNotFound, MultipleFound
from widedeep.ops.variables import savable_variables

import os
import re
import time
import numpy as np
from glob import glob

import tensorflow as tf

logger = get_logger('model')


""" Class that contains function helpers for network training """


def new_model(session):
    """ Initializes model from scratch and returns global step variable
    Args:
        session: Tensorflow session
    Returns:
        step: Global step variable
    """
    logger.info('Initializing model from scratch ...')
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    session.run(tf.initialize_all_tables())
    return get_global_step()


def restore_model(session, saver, path):
    """ Initializes a model that has been previously trained and
    returns global step
    Args:
        session: Tensorflow session
        saver: Tensorflow saver
        path: Path where model to be loaded is
    Returns:
        Global step variable
    """
    logger.info('Starting model from %s' % path)
    session.run(tf.local_variables_initializer())
    session.run(tf.initialize_all_tables())
    saver.restore(session, path)
    return get_global_step()


def get_session_config(gpu_mem, log_placement):
    """
    Args:
        gpu_mem: Percentage of GPU memory willing to use
        log_placement: Whether to log devices where variables are created
    Returns:
        config: Configuration related to GPU occupation from the configuration
    """
    gpu = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem)
    config = tf.ConfigProto(log_device_placement=log_placement,
                            allow_soft_placement=True,
                            gpu_options=gpu)
    return config


def _get_first_op_from_collection(collection_name):
    elements = tf.get_collection(collection_name)
    if elements:
        return elements[0]
    return None


def get_saver(max_keep=4):
    """Lazy init and return saver. """
    saver = _get_first_op_from_collection(tf.GraphKeys.SAVERS)
    if saver is None and tf.global_variables_initializer():
        saver = tf.train.Saver(max_to_keep=max_keep,
                               var_list=savable_variables())
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    return saver


def check_features(data):
    """ Checks input features follow the corresponding format """
    if not isinstance(data, dict):
        raise ValueError("Input data must be parsed from TFRecords "
                         + "and be provided as a dictionary")


def get_global_step(graph=None):
    """ Loads the unique global step, if found """
    graph = tf.get_default_graph() if graph is None else graph
    global_step_tensors = graph.get_collection(tf.GraphKeys.GLOBAL_STEP)
    if len(global_step_tensors) == 0:
        raise VariableNotFound('No global step stored in the collection')
    elif len(global_step_tensors) > 1:
        raise MultipleFound('Multiple instances found for the global step')
    return global_step_tensors[0]


def create_global_step():
    """ Creates a global step in the VARIABLEs and GLOBAL_STEP collections """
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
    return tf.get_variable('global_step', shape=[],
                           dtype=tf.int32,
                           initializer=tf.constant_initializer(0),
                           trainable=False,
                           collections=collections)


def create_execution_var():
    """ Creates a variable that is used to determine whether the execution
    trains or evaluates the graph """
    return tf.placeholder(tf.bool, name='is_training')


def initialize_queues(session):
    """ Initializes the defined queues in the session """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)
    return coord, threads


def finalize_queues(coord, threads):
    """ Finalized the queues used to enqueue examples """
    # When done, ask the threads to stop.
    coord.request_stop()
    # And wait for them to actually do it.
    coord.join(threads)


def prepare_logging(logs_path, graph):
    """ Merges summaries and creates writer and saver """
    saver = get_saver()
    summary_ops = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path, graph)
    return saver, summary_ops, writer


def log_network_step(step, loss, diff, tag, metrics={}):
    """ Logs the value of the input metrics
    Args:
        step: Current step
        loss: Loss value
        diff: Time the step has taken, in seconds
        tag: Tag to use for the dataset logged
        metrics: Dictionary of computed metrics
    """
    metrics_str = ', '.join([k + ': ' + str(v)
                             for (k, v) in list(metrics.items())])
    logger.info('%s step %d with loss %f (%f seconds). Metrics: %s'
                % (tag, step, loss, diff, metrics_str))


def store_summaries(writer, step, summary_str):
    """ Stores all summaries defined by the input operations so
    they can be visualized in Tensorboard """
    logger.info('Storing summaries at step %d' % step)
    writer.add_summary(summary_str, step)
    writer.flush()  # Needed sometimes


def store_checkpoint(sess, saver, step, folder):
    """ Stores a checkpoint of the model in the given session """
    logger.info('Storing model at step %d' % step)
    model_path = os.path.join(folder, 'model_' + str(step) + '.ckpt')
    saver.save(sess, model_path)


def get_models(folder):
    """
    Args:
        folder: Folder containing models
    Returns:
        Pairs of (step, path) containing the training steps and the meta path,
        sorted by the step
    """
    def get_num(s):
        return int(re.search('\d+', s).group(0))
    models = glob(os.path.join(folder, '*.ckpt.meta'))
    filenames = [os.path.basename(m) for m in models]
    pairs = [(get_num(m), m) for m in filenames]
    return sorted(pairs, key=lambda x: x[0])


def erase_model(folder, meta_file):
    """ Erases the files related to the model given by the input meta file """
    prefix = meta_file[:-5]
    files = glob(os.path.join(folder, prefix + '.*'))
    for f in files:
        logger.info('Erasing file %s ...' % f)
        os.remove(f)


def clean_models(folder, steps_keep=None):
    """ Clean models which from the folder, optinally keeping one of them
     Args:
         folder: Folder containing the models
         steps_keep: Model to keep. Set to None to erase all.
    """
    models = get_models(folder)
    for m in models:
        m_steps, m_path = m
        if steps_keep is None:
            erase_model(folder, m_path)
        else:
            if m_steps != steps_keep:
                erase_model(folder, m_path)


def keep_tracked_models(folder, track):
    """ Erases the checkpoint information keeping only the newest ones
     Args:
        folder: Folder containing the models
        track: Number of newest checkpoints to keep
    """
    models = get_models(folder)
    for m in models[:-track-1]:
        erase_model(folder, m[1])


class NetworkRun(object):

    def __init__(self, data, targets, train_ops, step_op, loss_op, logits_op,
                 is_training, metrics=[]):
        self._data_ops = data
        self._target_ops = targets
        self._train_ops = train_ops
        self._step_op = step_op
        self._loss_op = loss_op
        self._logits_op = logits_op
        self._is_training = is_training
        self._metrics = metrics

        self._loss_values = []
        self._metric_values = {}

    def _run(self, session, summary_ops, is_training):
        """ Peforms a single run according to the input settings
            Args:
                session: Tensorflow session
                summary_ops: Tensor of summary operations
                is_training: Whether we are updating the weights (True) or
                    evaluating the model (False)
            Returns:
                Run results
        """
        # Train and step operations are only sent on training runs
        train_ops = self._train_ops if is_training else []
        step_op = self._step_op if is_training else []

        # Gather metric operations
        metric_ops = [m.get_op() for m in self._metrics]

        # Arguments set in certain position
        run_args = [train_ops, step_op, self._loss_op, self._logits_op,
                    self._data_ops, self._target_ops] + metric_ops + \
                   [summary_ops]

        # Run operations, measure time lapse and gather results
        start = time.time()
        res = session.run(run_args, feed_dict={self._is_training: is_training})
        time_diff = time.time() - start

        # Parse returned values
        new_step, loss_value, rest = res[1], res[2], res[3:]
        logits_value, data, labels, sum_and_metrics = rest[0], rest[1], \
            rest[2], rest[3:]

        metric_values = sum_and_metrics[:-1]
        summary_str = sum_and_metrics[-1]

        # Build metric dictionary
        metrics_out = {
            m.get_label(): metric_values[i]
            for (i, m) in enumerate(self._metrics)
        }

        # Generate object gathering process output
        results = RunResults(loss_value=loss_value,
                             time_diff=time_diff,
                             metrics=metrics_out,
                             logits=logits_value,
                             data=data, labels=labels,
                             summary_str=summary_str,
                             new_step=new_step)

        # Update current state and return results
        self._update(results)
        return results

    def _update(self, results):
        """ Updates loss and metrics with the input results """
        self._loss_values.append(results.loss)
        for k, v in results.metrics.items():
            if k not in self._metric_values:
                self._metric_values[k] = []
            self._metric_values[k].append(v)

    def loss_average(self):
        """ Returns the average loss from all runs """
        return np.mean(self._loss_values)

    def metrics_average(self):
        """ Returns the metric average from all runs """
        results = {}
        for m in self._metrics:
            name = m.get_label()
            results[name] = m.reduce_mean(self._metric_values[name])
        return results

    def training_run(self, session, summary_ops, log=True):
        """ Runs a training step
        Args:
            session: Tensorflow session
            summary_ops: Summary operations for Tensorboard
            log: Whether to log step
        """
        run = self._run(session=session,
                        summary_ops=summary_ops,
                        is_training=True)
        if log:
            run.log('Training')
        return run

    def test_run(self, session, summary_ops, step, data_mode, log=True):
        """ Runs an evaluation step
        Args:
            session: Tensorflow session
            summary_ops: Summary operations for Tensorboard
            step: Step number to assign to the run
            data_mode: Evaluation mode (validation or testing)
            log: Whether to log step
        """
        run = self._run(session=session,
                        summary_ops=summary_ops,
                        is_training=False)
        run.step = step
        if log:
            run.log('Evaluation {}'.format(str(data_mode)))
        return run

    def manual_log(self, writer, step):
        """ Logs the loss and the output metrics so far """
        # Store loss value
        loss = self.loss_average()
        summary = tf.Summary(
            value=[tf.Summary.Value(tag='loss',
                                    simple_value=float(loss))]
        )
        writer.add_summary(summary, step)
        # Store metrics
        metrics_avg = self.metrics_average()
        for m in self._metrics:
            m.write_summary(metrics_avg[m.get_label()], writer, step)
        writer.flush()  # Needed sometimes


class RunResults(object):

    """ Gathers the results of running a training/evaluation step """

    def __init__(self, loss_value, time_diff, metrics, logits, data, labels,
                 summary_str, new_step=None):
        """ Initializes the values of the output of an optimization/evaluation
        step """
        self.loss = loss_value
        self.time = time_diff
        self.metrics = metrics
        self.logits = logits
        self.data = data
        self.labels = labels
        self.summary_str = summary_str
        self.step = new_step

    def log(self, prefix):
        log_network_step(self.step, self.loss, self.time, prefix, self.metrics)
