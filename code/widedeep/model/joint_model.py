from protodata.reading_ops import DataReader
import widedeep.model.model_utils as mu
from widedeep.model.model_base import CNNModel
from widedeep.ops.losses import LossFn, ClassificationLoss, RegressionLoss, \
    MixtureDensityLoss
import widedeep.utils as ut

from protodata.utils import create_dir
from protodata.data_ops import DataMode

import abc
import os
import numpy as np
import scipy.misc
import tensorflow as tf


logger = ut.get_logger('model')


class JointModel(object):
    """
    Model that trains multiple models independently by using a joint
    training loss. The output os each model is summed and loss is computed
    wrt each model independently using separate optimizers.

    Args:
        model_dir: Directory to save model parameters, graph and etc. This
            can also be used to load checkpoints from the directory into a
            estimator to continue training a previously saved model.
        models: List of models.
        outputs: Number of targets to predict.
        loss_fn: Loss function to use for training.
        l1_reg: L1 regularization. Set to None to disable it.
        l2_reg: L2 regularization. Set to None to disable it.
        clip_gradient: If not None, gradients are clipped using the given
            value as norm
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, model_dir, models, outputs, loss_fn, l1_reg=None,
                 l2_reg=None, clip_gradient=None):
        self._model_dir = model_dir
        create_dir(model_dir)
        self._models = models
        self._outputs = outputs
        self._l1_reg = l1_reg
        self._l2_reg = l2_reg
        self._clip_grad = clip_gradient

        # Check model names are unique
        if len(np.unique([m.get_name() for m in models])) != len(models):
            raise ValueError('Models provided should have unique names')

        # Log provided models
        for m in self._models:
            logger.info('Provided model "%s"' % m.get_name())

        if not isinstance(loss_fn, LossFn):
            raise ValueError("Loss must be a subclass of LossFn")
        else:
            self._loss_fn = loss_fn

    def _get_logits(self, features, is_training):
        """ Returns the joint output of all the models before  """
        if not self._models:
            raise RuntimeError('No models in the joint model.'
                               + 'Must have at least one model')
        num_outs = self.get_output_num()
        return tf.add_n([m.get_logits(features,
                                      is_training=is_training,
                                      outputs=num_outs)
                        for m in self._models])

    def _load_pretrained(self):
        """ Calls model-specific functions for weight assignation """
        all_ops = []
        for m in self._models:
            all_ops += m.warm_start()
        return all_ops

    def train(self,
              dataset,
              batch_size,
              steps=None,
              max_steps=None,
              metrics=[],
              track_summaries=1,
              track_models=1,
              gpu_frac=1.0,
              workers=4,
              mem_dequeing=2,
              log_steps=10):
        """ Starts training for the requested iterations
        Args:
            dataset: Data reading settings
            batch_size: Number of rows per batch
            steps: Steps to train
            max_steps: Maximum number of steps in training
            metrics: Metrics to use for the network evaluation
            track_summaries: Steps between Tensorboard tracked summaries.
                Set to None to disable
            track_models: Steps between auxiliar model checkpoints.
                Set to None to disable
            gpu_frac: Ratio of GPU memory to use. Only used if GPU available
            workers: CPU workers to use for dequeuing examples
            mem_dequeing: Memory factor (~GB) to use for dequeing
            log_steps: Steps between logging of training steps.
        Returns
            step: Step at which training has stopped
            loss: Mean loss in the process
            metrics: Mean metrics values in the process
        """
        if steps is None and max_steps is None:
            raise ValueError('Either number of steps or maximum allowed steps'
                             + 'should be provided (or both)')

        return self._train_network(dataset,
                                   logs_path=self._model_dir,
                                   batch_size=batch_size,
                                   steps=steps,
                                   max_steps=max_steps,
                                   metrics=metrics,
                                   track_summaries=track_summaries,
                                   track_models=track_models,
                                   gpu_frac=gpu_frac,
                                   workers=workers,
                                   mem_dequeing=mem_dequeing,
                                   log_steps=log_steps)

    def evaluate(self, dataset, data_mode, batch_size, track_summaries=1,
                 metrics=[], gpu_frac=0.8, workers=2, mem_dequeing=2):
        """ Evaluates the whole input dataset
        Args:
            See _eval_network arguments.
        Returns
            See _eval_network returns.
        """
        folder_name = '_'.join([str(data_mode).lower(),
                                str(ut.current_time_id())])
        out_path = os.path.join(self._model_dir, folder_name)
        return self._eval_network(dataset,
                                  logs_path=out_path,
                                  batch_size=batch_size,
                                  data_mode=data_mode,
                                  metrics=metrics,
                                  track_summaries=track_summaries,
                                  gpu_frac=gpu_frac,
                                  workers=workers,
                                  mem_dequeing=mem_dequeing)

    def train_and_validate(self,
                           dataset,
                           batch_size,
                           validate_steps,
                           validate_interval,
                           patience=2,
                           steps=None,
                           max_steps=None,
                           track_summaries=1,
                           metrics=[],
                           gpu_frac=1.0,
                           workers=1,
                           mem_dequeing=3,
                           keep_models=True,
                           log_steps=10):
        """ Trains the network using validation and early stop. Training stops
        when the number of successive increases of validation loss reaches the
        patience value.
        Args:
            See train_network for other arguments.
            validate_steps: Number of validation steps.
            validate_interval: Training steps between validation phases.
            patience: Maximum successive number of validation loss increase
                that we want to allow.
            keep_models: Whether to keep the best model or erase it to save
                space. Default is True
        Returns
            See train_network for returns.
        """

        if (steps is not None and validate_interval > steps) or \
                (max_steps is not None and validate_interval > max_steps):
            raise ValueError('Number of steps must be bigger than validation' +
                             ' interval')

        # Create folder for training
        training_path = os.path.join(self._model_dir, 'training')
        create_dir(training_path)

        # Create folder for validation
        validation_path = os.path.join(self._model_dir, 'validation')
        create_dir(validation_path)

        # Training outer loop
        initial_counter = None
        consecutive, counter = 0, 0
        best = (0, np.inf)
        end = False
        while not end and consecutive < patience:

            # Adjust training steps
            if steps is not None and initial_counter is not None \
                    and counter + validate_interval > steps:
                train_steps = steps - counter
                end = True
            else:
                train_steps = validate_interval

            if not end:

                # Train network
                step, _, _ = self._train_network(
                    dataset,
                    logs_path=self._model_dir,
                    batch_size=batch_size,
                    steps=train_steps,
                    max_steps=max_steps,
                    metrics=metrics,
                    track_summaries=track_summaries,
                    track_models=None,
                    gpu_frac=gpu_frac,
                    workers=workers,
                    mem_dequeing=mem_dequeing,
                    log_steps=log_steps
                )

                # Set initial step when starting training
                if initial_counter is None:
                    initial_counter = step - train_steps

                # Validate using training
                _, _ = self._eval_network(dataset,
                                          logs_path=training_path,
                                          batch_size=batch_size,
                                          data_mode=DataMode.TRAINING,
                                          metrics=metrics,
                                          steps=validate_steps,
                                          gpu_frac=gpu_frac,
                                          workers=workers,
                                          mem_dequeing=mem_dequeing)

                # Validate using validation
                val_loss, _ = self._eval_network(dataset,
                                                 logs_path=validation_path,
                                                 batch_size=batch_size,
                                                 data_mode=DataMode.VALIDATION,
                                                 metrics=metrics,
                                                 steps=validate_steps,
                                                 gpu_frac=gpu_frac,
                                                 workers=workers,
                                                 mem_dequeing=mem_dequeing)

                # Check if max step reached
                if max_steps is not None and step >= max_steps:
                    end = True

                # Check validation loss hasn't decreased
                if val_loss > best[1]:
                    logger.warn('Validation loss at step %d (%f) is worse than'
                                % (step, val_loss) +
                                'best (%f) at step %d for %d time/s in a row. '
                                % (best[1], best[0], consecutive))
                    consecutive += 1
                else:
                    logger.warn('Validation results improved at step' +
                                '%d.Previous best was %f at step %d and'
                                % (step, best[1], best[0]) +
                                'new new best is %f.' % (val_loss))
                    consecutive = 0
                    best = (step, val_loss)

                # Clear old models
                mu.keep_tracked_models(self._model_dir, patience)

                # Update counts
                counter = step - initial_counter

        best_step, best_loss = best
        logger.info('Best model found was at step %d with loss %f'
                    % (best_step, best_loss))

        # Clean folder. If keep models is True keep best
        #  Otherwise keep no model
        keep_model = best_step if keep_models is True else None
        mu.clean_models(self._model_dir, keep_model)

        return best

    def _initialize_model(self, sess, saver, is_training):
        """ Loads the model if a trained model is available or
        starts one from scratch

        Args:
            sess: Tensorflow session
            saver: Tensorflow saver
            is_training: Whether we are training or evaluating

        Returns:
            step: Current step of the model
            start: Whether the model is started from scratch
        """
        last_model = tf.train.latest_checkpoint(self._model_dir)

        if last_model is None and is_training is False:
            raise ut.ModelNotFound('Could not find model in the directory')

        if last_model is None:
            logger.info('Initializing model from scratch ...')
            step, start = mu.get_global_step(), True
        else:
            logger.info('Loading model from %s ...' % last_model)
            saver.restore(sess, last_model)
            step, start = mu.get_global_step(), False

        return sess.run(step), start

    def _train_network(self,
                       dataset,
                       logs_path,
                       batch_size,
                       metrics,
                       track_summaries,
                       gpu_frac,
                       workers,
                       mem_dequeing,
                       track_models=None,
                       steps=None,
                       max_steps=None,
                       log_steps=10):
        """ Runs training on the network defined for the given number of steps
        and stores a checkpoint at the end.
        Args:
            See train_network for other arguments.
            logs_path: Path where to store the network stats
        Returns
            step: Step at which training has stopped
            loss: Mean loss in the process
            metrics: Mean metrics values in the process
        """
        # Create reader to get TFRecords
        reader = DataReader(dataset)

        with tf.Graph().as_default() as g:

            # Read data and define operations
            data, labels = reader.read_batch(batch_size,
                                             DataMode.TRAINING,
                                             mem_dequeing,
                                             workers)
            main_run = self._build_run_settings(data, labels, metrics)

            # Load weights ops for warm start on training
            assign_ops = self._load_pretrained()

            # Prepare logging for Tensorboard
            saver, summary_ops, writer = mu.prepare_logging(logs_path, g)

            # Supervisor for training. We only want it to deal with the
            # session. Initializes variables
            supervisor = tf.train.Supervisor(graph=g)

            # Initialize session
            session_conf = mu.get_session_config(gpu_frac, log_placement=False)
            with supervisor.managed_session(config=session_conf) as sess:

                # Load model, if existing. Otherwise start from scratch
                step_value, start = self._initialize_model(sess,
                                                           saver=saver,
                                                           is_training=True)

                # Get stopping condition according to mode
                step_limit = step_value + steps if steps is not None else max_steps  # noqa
                step_limit = step_limit if max_steps is None else min(step_limit, max_steps)  # noqa
                stop = step_value >= step_limit

                # Assign weights only if model started from scratch
                if start is True and len(assign_ops) > 0:
                    logger.info('Assigning pretrained values to models ...')
                    sess.run(assign_ops)

                try:
                    # Let queues start dequeing examples
                    coord, threads = mu.initialize_queues(sess)

                    while not stop:

                        # Run network
                        log_run = step_value % log_steps == 0
                        main_res = main_run.training_run(sess,
                                                         summary_ops,
                                                         log=log_run)

                        # Track summaries if needed
                        if track_summaries is not None and \
                                step_value % track_summaries == 0:
                            mu.store_summaries(writer,
                                               step_value,
                                               main_res.summary_str)

                        # Track models if needed
                        if track_models is not None \
                                and step_value % track_models == 0 \
                                and step_value != 0:
                            mu.store_checkpoint(sess,
                                                saver,
                                                step_value,
                                                logs_path)

                        # Update current step and stop condition
                        step_value = main_res.step
                        stop = step_value >= step_limit

                    # Store model at exit
                    mu.store_checkpoint(sess, saver, step_value, logs_path)

                    try:
                        mu.finalize_queues(coord, threads)
                    except RuntimeError as e:
                        logger.warning('Error stopping coordinator: %s', e)

                except tf.errors.OutOfRangeError as e:
                    logger.warn('Input queue exhausted due to ' +
                                'unexpected reason: %s.', e)

                return step_value, main_run.loss_average(), main_run.metrics_average()  # noqa

    def _eval_network(self, dataset, data_mode, logs_path, batch_size, metrics,
                      gpu_frac, workers, mem_dequeing, track_summaries=50,
                      steps=None):
        """ Runs and tracks a single evaluation step on the given data subset
        Args:
            Other arguments can be seen in train_network
            data_mode: Data subset to use for evaluation
            track_summaries: Steps between Tensorboard summaries.
                Only used if steps is None
            steps: Number of batches the model is evaluated on.
                If not None, a single summary is created at the end.
                If None, the model is evaluated on the whole dataset and it
                tracks a summary periodically.
        Returns
            loss: Loss produced by the batch
            metrics: Set of metrics values
        """
        # Create reader to get TFRecords
        reader = DataReader(dataset)

        with tf.Graph().as_default() as g:

            # Read data and define operations
            data, labels = reader.read_batch(batch_size,
                                             data_mode,
                                             mem_dequeing,
                                             workers,
                                             shuffle=True,
                                             train_mode=False)

            main_run = self._build_run_settings(data, labels, metrics)

            # Prepare logging for Tensorboard
            saver, summary_ops, writer = mu.prepare_logging(logs_path, g)

            # Supervisor for training. We only want it to deal with the
            # session. Initializes variables
            supervisor = tf.train.Supervisor(graph=g)

            # Initialize session
            session_conf = mu.get_session_config(gpu_frac, log_placement=False)
            with supervisor.managed_session(config=session_conf) as sess:

                # Load model, if existing. Otherwise start from scratch
                step_value, _ = self._initialize_model(sess,
                                                       saver=saver,
                                                       is_training=False)

                try:
                    # Let queues start dequeing examples
                    coord, threads = mu.initialize_queues(sess)

                    # Initialize loop conditions
                    step_counter = 0
                    stop = False if steps is None else step_counter >= steps

                    while not stop:

                        # Run evaluation
                        res = main_run.test_run(sess,
                                                summary_ops,
                                                step_value,
                                                data_mode=data_mode,
                                                log=True)

                        if steps is not None and steps == step_counter:
                            # Reached max steps, store summary and stop
                            main_run.manual_log(writer, step_value)
                            stop = True
                        elif steps is None \
                                and step_counter % track_summaries == 0:
                            # Periodic storage of summaries
                            mu.store_summaries(writer,
                                               step_counter,
                                               res.summary_str)

                        step_counter += 1

                    try:
                        mu.finalize_queues(coord, threads)
                    except RuntimeError as e:
                        logger.warning('Error stopping coordinator: %s', e)

                except tf.errors.OutOfRangeError:
                    logger.info('Queue run out of evaluation instances')

                return main_run.loss_average(), main_run.metrics_average()

    def eval_places(self,
                    dataset,
                    data_mode,
                    gpu_frac,
                    workers,
                    mem_dequeing,
                    steps=20):
        """ Runs and tracks a single evaluation step on the given data subset
        Args:
            Other arguments can be seen in _train_network
            data_mode: Data subset to use for evaluation
            steps: Number of examples the model is evaluated on.
        """
        # Create reader to get TFRecords
        reader = DataReader(dataset)

        with tf.Graph().as_default() as g:

            # Read data and define operations
            data, labels = reader.read_batch(1,
                                             data_mode,
                                             mem_dequeing,
                                             workers,
                                             shuffle=True,
                                             train_mode=False)

            main_run = self._build_run_settings(data, labels, [])

            # Load weights ops for warm start on training
            assign_ops = self._load_pretrained()

            # Supervisor for training. We only want it to deal with the
            # session. Initializes variables
            supervisor = tf.train.Supervisor(graph=g)

            # Initialize session
            session_conf = mu.get_session_config(gpu_frac,
                                                 log_placement=False)
            with supervisor.managed_session(config=session_conf) as sess:

                # Load model, if existing. Otherwise start from scratch
                step_value, _ = self._initialize_model(sess,
                                                       saver=mu.get_saver(),
                                                       is_training=True)

                if len(assign_ops) > 0:
                    logger.info('Assigning pretrained values to models ...')
                    sess.run(assign_ops)
                else:
                    raise RuntimeError("Unexpected error: no assign operations"
                                       + " to load weights")

                try:
                    # Let queues start dequeing examples
                    coord, threads = mu.initialize_queues(sess)
                    counter = 0
                    while counter < steps:
                        self.test_places_output(sess, data, main_run)
                        counter += 1

                    try:
                        mu.finalize_queues(coord, threads)
                    except RuntimeError as e:
                        logger.warning('Error stopping coordinator: %s', e)

                except tf.errors.OutOfRangeError:
                    logger.info('Queue run out of evaluation instances')

    def _build_run_settings(self, data, labels, metrics):
        """ Builds a wrapper to easily run the operations """
        # Create training indicator and global step
        exec_mode, step = mu.create_execution_var(), mu.create_global_step()
        # Define model operations
        train_ops, logits, loss_op, metrics, step_op = \
            self.get_model_ops(step=step,
                               data=data,
                               labels=labels,
                               metrics=metrics,
                               is_training=exec_mode)

        # Return run settings
        return mu.NetworkRun(data=data,
                             targets=labels,
                             train_ops=train_ops,
                             step_op=step_op,
                             loss_op=loss_op,
                             metrics=metrics,
                             logits_op=logits,
                             is_training=exec_mode)

    def get_model_ops(self, step, data, labels, metrics, is_training):
        """ Returns the main operation to send to execution for each
        training/testing step

           Args:
               step: Global step variable.
               data: Input data.
               labels: Groundtruth corresponding to outputs.
               metrics: Metrics to compute at evaluation.
               is_training: Boolean 1D tensor indicating whether the operation
                trains (True) or evaluates (False)

           Returns:
               train_op: Training operation needed to update the model
                    (just used if training)
               loss_op: Loss function to compute
               metrics: Updated metrics with the tensorflow operations.
       """
        # Check input format and compute joint loss by combining specific
        mu.check_features(data)
        logits = self._get_logits(data, is_training=is_training)

        # Get training loss and metrics
        loss_op = self._loss_fn.compute_loss(logits, labels)

        # Track loss after regularization
        tf.summary.scalar(name='raw_loss', tensor=loss_op)

        # Add L1 regularization, if requested
        if self._l1_reg is not None:
            loss_op = self.add_l1_regularization(loss_op)

        # Add L2 regularization, if requested
        if self._l2_reg is not None:
            loss_op = self.add_l2_regularization(loss_op)

        # Compute metric operations
        for m in metrics:
            predicted = self.predict(logits)
            m.predicted = predicted
            m.compute_op(predicted, labels)

        # Track loss after regularization
        tf.summary.scalar(name='loss', tensor=loss_op)

        # Update all operations
        extra_updates = self._extra_update_ops()
        if len(extra_updates) > 0:
            with tf.control_dependencies(extra_updates):
                train_ops = self._update_operations(step,
                                                    loss_op,
                                                    self._clip_grad)
        else:
            train_ops = self._update_operations(step, loss_op, self._clip_grad)

        # Increase step
        step_op = step.assign(step + 1)

        return train_ops, logits, loss_op, metrics, step_op

    def add_l1_regularization(self, loss_op):
        """ Adds L1 regularization to the input loss """
        logger.info("Enabling L1 regularization")
        l1_term = tf.multiply(self._l1_reg, self._get_l1_norm())
        tf.summary.scalar(name='l1_term', tensor=l1_term)
        return loss_op + l1_term

    def add_l2_regularization(self, loss_op):
        """ Adds L2 regularization to the input loss """
        logger.info("Enabling L2 regularization")
        l2_term = tf.multiply(self._l2_reg, self._get_l2_norm())
        tf.summary.scalar(name='l2_term', tensor=l2_term)
        return loss_op + l2_term

    def _get_l1_norm(self):
        """ Returns the L1 norm of the weights of the models """
        return tf.add_n([tf.reduce_sum(tf.abs(x))
                        for x in self._get_model_weights()])

    def _get_l2_norm(self):
        """ Returns the L2 norm of the weights of the models """
        return tf.add_n([tf.nn.l2_loss(x) for x in self._get_model_weights()])

    def _get_model_weights(self):
        """ Returns the list of models parameters """
        weights = []
        for m in self._models:
            for p in m.get_parameters():
                if 'bias' in p.name:
                    logger.info('Ignoring bias %s for regularization ("%s")'
                                % (p.name, m.get_name()))
                elif 'weight' in p.name:
                    weights.append(p)
                    logger.info('Using weights %s for regularization ("%s")'
                                % (p.name, m.get_name()))
                else:
                    logger.info('Ignoring unknown parameter type %s'
                                % (p.name) +
                                'for regularization ("%s")'
                                % (m.get_name()))

        return weights

    def _extra_update_ops(self):
        """ Return the additional graph for optimization in the model """
        ops = self._models[0].get_extra_operations()
        for i in range(1, len(self._models)):
            ops = ops + self._models[i].get_extra_operations()
        return ops

    def _update_operations(self, step, loss, clip_grad):
        """ Return the training graph operations of the models in the model """
        return [m.get_training_ops(step, loss, clip_grad)
                for (i, m) in enumerate(self._models)]

    def test_places_output(self, sess, data, run):
        """ Tests the Alexnet places weight pretraining by showing the top scores
        of a given image """

        def read_places():
            with open(ut.get_places365_categories()) as f:
                lines = f.readlines()
            d = {}
            for i in lines:
                name, id = i.strip().split(' ')
                d[int(id)] = name
            return d

        # Get first cnn model in the joint model
        cnn_models = [m for m in self._models if isinstance(m, CNNModel)]

        if len(cnn_models) == 0:
            raise RuntimeError('No CNN model found')

        # Read data from session
        places_dict = read_places()
        feed_args = {run._is_training: False}
        places_output, data_out = sess.run([cnn_models[0].last_hidden, data],
                                           feed_dict=feed_args)

        # Convert to list and assign id
        index = 0
        places_out_list = places_output[index].tolist()

        # Get top tags
        zipped_places = list(zip(range(len(places_out_list)),
                                 places_out_list))
        zipped_places.sort(key=lambda t: t[1], reverse=True)

        # Display top 5 tags
        for (id, score) in zipped_places[:5]:
            name = places_dict[id]
            logger.info('Tag %s found with score %f' % (name, score))

        # Display last image in batch
        img = np.squeeze(data_out['image'][index])
        img = img[..., ::-1]  # To RGB
        scipy.misc.imshow(img)

    @abc.abstractmethod
    def get_output_num(self):
        """ Returns the number of outputs of the model """

    @abc.abstractmethod
    def predict(self, logits):
        """ Returns the prediction of the model given the model output """


class NeuralNetwork(JointModel):

    """ Traditional Feedforward neural network """

    def get_output_num(self):
        return self._outputs

    def predict(self, logits):
        return logits


class JointClassifier(NeuralNetwork):
    """ Joint deterministic classifier. See base class """

    def __init__(self,
                 model_dir,
                 models,
                 outputs,
                 l1_reg,
                 l2_reg,
                 loss_fn,
                 clip_gradient):

        super(JointClassifier, self).__init__(
            model_dir=model_dir,
            models=models,
            outputs=outputs,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            loss_fn=loss_fn,
            clip_gradient=clip_gradient
        )

        # At this point we know all models have the same number of classes
        if outputs < 2:
            raise ValueError(
                'Classification models should have at least 2 classes'
            )

        if not isinstance(loss_fn, ClassificationLoss):
            raise ValueError(
                'Classification problems must have a classification loss'
            )


class JointRegressor(NeuralNetwork):
    """ Joint deterministic regressor. See base class """

    def __init__(self,
                 model_dir,
                 models,
                 outputs,
                 l1_reg,
                 l2_reg,
                 loss_fn,
                 clip_gradient):

        super(JointRegressor, self).__init__(
            model_dir=model_dir,
            models=models,
            outputs=outputs,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            loss_fn=loss_fn,
            clip_gradient=clip_gradient
        )

        if not isinstance(loss_fn, RegressionLoss):
            raise ValueError(
                'Regression problems must have a classification loss'
            )


class JointMDN(JointModel):

    """ Joint mixture density model where outputs are predicted using a
    mixture model composed by several gaussian components """

    def __init__(self,
                 mixture,
                 model_dir,
                 models,
                 outputs,
                 l1_reg,
                 l2_reg,
                 clip_gradient):
        """
        Args:
            mixture: Mixture distribution to use (e.g. GaussianMixture)
            For other arguments check JointModel constructir
        """
        self.mixture = mixture
        super(JointMDN, self).__init__(
            model_dir=model_dir,
            models=models,
            outputs=outputs,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            loss_fn=MixtureDensityLoss(distribution=mixture)
        )

    def get_output_num(self):
        return self.mixture.outputs(self._outputs)

    def predict(self, logits):
        return self.mixture.sample(logits)
