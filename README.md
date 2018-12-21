# Practical-Guide

## verbose
verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

## ModelCheckpoint
ModelCheckpoint((filepath, monitor='val_loss', 
                 verbose=0, save_best_only=False, 
                 save_weights_only=False, mode='auto', 
                 period=1)):
                 
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    
    
    
## EarlyStopping
EarlyStopping(monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False)
                 
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """




## TensorBoard
TensorBoard(log_dir='./logs',
             histogram_freq=0,
             batch_size=32,
             write_graph=True,
             write_grads=False,
             write_images=False,
             embeddings_freq=0,
             embeddings_layer_names=None,
             embeddings_metadata=None,
             embeddings_data=None,
             update_freq='epoch')
             
    """TensorBoard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/guide/embedding#metadata)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings](
            https://www.tensorflow.org/guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    """





## ReduceLROnPlateau
ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, 
                 min_lr=0,**kwargs)
                 
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    # Example
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """




## fit(self,
        fit(self
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):
        """Trains the model for a given number of epochs (iterations on a dataset).
        # Arguments
            x: Numpy array of training data (if the model has a single input),
                or list of Numpy arrays (if the model has multiple inputs).
                If input layers in the model are named, you can also pass a
                dictionary mapping input names to Numpy arrays.
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: Numpy array of target (label) data
                (if the model has a single output),
                or list of Numpy arrays (if the model has multiple outputs).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
            validation_data: tuple `(x_val, y_val)` or tuple
                `(x_val, y_val, val_sample_weights)` on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
            validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.
        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        # Raises
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        
        
        
        
 ## fit_generator
 
     def fit_generator(self, generator,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0):
        """Trains the model on data generated batch-by-batch by a Python generator
        (or an instance of `Sequence`).
        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.
        The use of `keras.utils.Sequence` guarantees the ordering
        and guarantees the single use of every input per epoch when
        using `use_multiprocessing=True`.
        # Arguments
            generator: A generator or an instance of `Sequence`
                (`keras.utils.Sequence`) object in order to avoid
                duplicate data when using multiprocessing.
                The output of the generator must be either
                - a tuple `(inputs, targets)`
                - a tuple `(inputs, targets, sample_weights)`.
                This tuple (a single output of the generator) makes a single
                batch. Therefore, all arrays in this tuple must have the same
                length (equal to the size of this batch). Different batches may
                have different sizes. For example, the last batch of the epoch
                is commonly smaller than the others, if the size of the dataset
                is not divisible by the batch size.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                batches have been seen by the model.
            steps_per_epoch: Integer.
                Total number of steps (batches of samples)
                to yield from `generator` before declaring one epoch
                finished and starting the next epoch. It should typically
                be equal to the number of samples of your dataset
                divided by the batch size.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire data provided,
                as defined by `steps_per_epoch`.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_data: This can be either
                - a generator or a `Sequence` object for the validation data
                - tuple `(x_val, y_val)`
                - tuple `(x_val, y_val, val_sample_weights)`
                on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
            validation_steps: Only relevant if `validation_data`
                is a generator. Total number of steps (batches of samples)
                to yield from `validation_data` generator before stopping
                at the end of every epoch. It should typically
                be equal to the number of samples of your
                validation dataset divided by the batch size.
                Optional for `Sequence`: if unspecified, will use
                the `len(validation_data)` as a number of steps.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only). This can be useful to tell the model to
                "pay more attention" to samples
                from an under-represented class.
            max_queue_size: Integer. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Maximum number of processes to spin up
                when using process-based threading.
                If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
            use_multiprocessing: Boolean.
                If `True`, use process-based threading.
                If unspecified, `use_multiprocessing` will default to `False`.
                Note that because this implementation
                relies on multiprocessing,
                you should not pass non-picklable arguments to the generator
                as they can't be passed easily to children processes.
            shuffle: Boolean. Whether to shuffle the order of the batches at
                the beginning of each epoch. Only used with instances
                of `Sequence` (`keras.utils.Sequence`).
                Has no effect when `steps_per_epoch` is not `None`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        # Raises
            ValueError: In case the generator yields data in an invalid format.
        # Example
        ```python
        def generate_arrays_from_file(path):
            while True:
                with open(path) as f:
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
        ```
        """
        
        
        
        
        
## Custorm Training        

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    earlystopper = EarlyStopping(patience=25, verbose=1)
    checkpointer = ModelCheckpoint('recsys.h5', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    rec_sys.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['mae','accuracy'])


    def rec_sys_data(dataset=data, batch_size=batch):
        while True:
            users, problems, labels, _, _, _ = generate_dataset(dataset=data, batch_size=batch_size)

            yield [users, problems], labels


    valid_users, valid_problems, valid_labels, _, _, _ = generate_dataset(dataset=data, batch_size=4)


    # # we want a constant validation group to have a frame of reference for model performance
    # valid_a, valid_b, valid_sim = gen_random_batch(count_list, all_dir, 32)
    batch=4
    loss_history = rec_sys.fit_generator(rec_sys_data(dataset=data, batch_size=batch), 
                                    steps_per_epoch = 100,#500,
                                    validation_data=([valid_users, valid_problems], valid_labels),
                                    epochs = 10,
                                    verbose = True,
                                    callbacks=[earlystopper,checkpointer])# make a generator out of the data








## Stratified Sampling with Keras ==>> Idea is to take one Fold and reset the model and train it again

    from sklearn.model_selection import StratifiedKFold

    # Instantiate the cross validator
    skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        print "Training on fold " + str(index+1) + "/10..."

        # Generate batches from indices
        xtrain, xval = X[train_indices], X[val_indices]
        ytrain, yval = y[train_indices], y[val_indices]

        # Clear model, and create it
        model = None
        model = create_model()

        # Debug message I guess
        # print "Training new iteration on " + str(xtrain.shape[0]) + " training samples, " + str(xval.shape[0]) + " validation samples, this may be a while..."

        history = train_model(model, xtrain, ytrain, xval, yval)
        accuracy_history = history.history['acc']
        val_accuracy_history = history.history['val_acc']
        print "Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1])
        
        
        
        
        
        
        
## stratify sampling or splitting       
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.col1,df.target,
                                                        stratify=df.target, 
                                                        test_size=0.2)     
