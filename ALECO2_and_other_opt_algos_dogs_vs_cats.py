from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, Callback
from keras import applications

import h5py
from sklearn.utils import shuffle


from keras.optimizers import Optimizer
from keras import backend as K
from legacy import interfaces
from keras.optimizers import SGD

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops


class MyEarlyStopping(Callback):

    def __init__(self, monitor='loss', factor=0.1, patience=1,
                 verbose=1, mode='auto', epsilon=1e-4, cooldown=0, min_xi=0):
        super(MyEarlyStopping, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('MyEarlyStopping '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_xi = min_xi
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Xi Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.xi_epsilon = self.min_xi * 1e-4

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce xi on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:

            IFF = float(K.get_value(self.model.optimizer.IFF))
            IGG = float(K.get_value(self.model.optimizer.IGG))
            IGF = float(K.get_value(self.model.optimizer.IGF))
            xi = float(K.get_value(self.model.optimizer.xi))

            if xi-1.0 < 0.0000000000001:
                print('\nEpoch %d: xi is %f' % (epoch, xi))
                self.model.stop_training = True
                print("\nEpoch %d: early stopping training" % epoch)

            if self.verbose > 0:
                print('\nEpoch %d: quantity is %g' % (epoch,IFF*IGG-IGF*IGF))

    def in_cooldown(self):
        return self.cooldown_counter > 0

class ReducedPOnPlateau(Callback):
    """Reduce dP when a metric has stopped improving.
    Models often benefit from reducing dP by a factor
    of 1-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, dP is reduced.
    # Example
        ```python
        reduce_dP = ReducedPOnPlateau(monitor='val_loss', factor=0.1,
                                      patience=5, min_dP=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_dP])
        ```
    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which dP will
            be reduced. new_dP = dP * factor
        patience: number of epochs with no improvement
            after which dP will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            dP will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after dP has been reduced.
        min_dP: lower bound on dP.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=1, mode='auto', epsilon=1e-4, cooldown=0, min_dP=0):
        super(ReducedPOnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReducedPOnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_dP = min_dP
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('dP Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.dP_epsilon = self.min_dP * 1e-4

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['dP'] = K.get_value(self.model.optimizer.dP)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce dP on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_dP = float(K.get_value(self.model.optimizer.dP))
                    if old_dP > self.min_dP + self.dP_epsilon:
                        new_dP = old_dP * self.factor
                        new_dP = max(new_dP, self.min_dP)
                        K.set_value(self.model.optimizer.dP, new_dP)
                        if self.verbose > 0:
                            print('\nEpoch %05d: reducing dP radius to %s.' % (epoch, new_dP))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0

class ReduceXiOnPlateau(Callback):
    """Reduce Xi when a metric has stopped improving.
    Models often benefit from reducing xi by a factor
    of 1-1000 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, xi is reduced.
    # Example
        ```python
        reduce_xi = ReduceXiOnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_xi=1e-5)
        model.fit(X_train, Y_train, callbacks=[reduce_xi])
        ```
    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which xi will
            be reduced. new_xi = xi * factor
        patience: number of epochs with no improvement
            after which xi will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            xi will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after xi has been reduced.
        min_xi: lower bound on xi.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=1, mode='auto', epsilon=1e-4, cooldown=0, min_xi=0):
        super(ReduceXiOnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceXiOnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_xi = min_xi
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('xi Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.xi_epsilon = self.min_xi * 1e-4

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['xi'] = K.get_value(self.model.optimizer.xi)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce xi on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    old_xi = float(K.get_value(self.model.optimizer.xi))
                    print('\nEpoch %05d: reducing xi angle to %s.' % (epoch, old_xi))
                    if old_xi > self.min_xi + self.xi_epsilon:
                        new_xi = old_xi * self.factor
                        new_xi = max(new_xi, self.min_xi)
                        K.set_value(self.model.optimizer.xi, new_xi)
                        if self.verbose > 0:
                            print('\nEpoch %05d: reducing xi angle to %s.' % (epoch, new_xi))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0

class ALECO2(Optimizer):
    """ALECO2 belongs to the class of Algorithms for Learning Efficiently with
    Constrained Optimization (ALECO) and this version (2) uses Adaptive Momentum

    Includes support for dP decay, and xi decay.

    # Arguments
        dP: float > 0. The algorithm works better if 0 < dP < 0.3
        xi: float > 0 and < 1. The algorithm works better if 0.6 < xi < 1
    """

    def __init__(self, dP=0.01, xi=0.85, decay=0.0, **kwargs):
        super(ALECO2, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.dP = K.variable(dP, name='dP')
        self.xi = K.variable(xi, name='xi')
        self.decay = K.variable(decay, name='decay')

        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):

        grads = K.gradients(loss, params)
        flattenedgrads = [K.flatten(x) for x in grads]
        G = K.concatenate(flattenedgrads)
        self.updates = []

        dP = self.dP
        xi = self.xi

        if self.initial_decay > 0:
            dP *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]

        flattenedmoments = [K.flatten(x) for x in moments]
        F = K.concatenate(flattenedmoments)

        self.weights = [self.iterations] + moments

        IGG=K.sum(G * G)
        IFF=K.sum(F * F)
        IGF=K.sum(G * F)
        dQ=-xi*dP*K.sqrt(IGG)
        lamda2= 0.5*K.sqrt((IFF*IGG-IGF*IGF)/(IGG*dP*dP-dQ*dQ))
        lamda1=(-2*lamda2*dQ+IGF)/IGG

        for p, g, m in zip(params, grads, moments):

            cond=K.greater(IFF,0.0)

            v = K.switch(cond, -((lamda1/(2*lamda2))*g)+((1/(2*lamda2))*m), -dP * g)

            self.updates.append(K.update(m, v))

            new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates


    def get_config(self):
        config = {'dP': float(K.get_value(self.dP)),
                  'xi': float(K.get_value(self.xi)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(ALECO2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#-----------------------------------------------------------------------------#


path = '/output/dogscats/'
batch_size = 32

X_train = []
X_valid = []

h5_filename='/dogs-vs-cats-redux/output/dogscats/gap_ResNet502017_09_17_08_44_42.h5'

for filename in [h5_filename]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_valid.append(np.array(h['valid']))
        y_train = np.array(h['label'])
        y_valid = np.array(h['val_label'])

X_train = np.concatenate(X_train, axis=1)
X_valid = np.concatenate(X_valid, axis=1)

X_train, y_train = shuffle(X_train, y_train)

# **Reduce learning rate when 'val_loss' has stopped improving.**

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

# **Reduce dP or xi when 'val_acc' has stopped improving.**

reduce_dP = ReducedPOnPlateau(monitor='val_acc', factor=0.2, patience=5, min_dP=1e-5)
reduce_xi = ReduceXiOnPlateau(monitor='val_acc', factor=0.2, patience=5, min_xi=1e-5)


input_tensor = Input(X_train.shape[1:])
x = input_tensor
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# **Optimizer 1: RMSprop**

model_rmsprop = Model(input_tensor, x)
model_rmsprop.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
hist_rmsprop = model_rmsprop.fit(X_train, y_train, batch_size=batch_size*2,
                  epochs=1, validation_data=(X_valid,y_valid), verbose=0, callbacks=[reduce_lr])


# **Optimizer 2: Adam**

model_adam = Model(input_tensor, x)
model_adam.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
hist_adam = model_adam.fit(X_train, y_train, batch_size=batch_size*2,
               epochs=300, validation_data=(X_valid,y_valid), verbose=0, callbacks=[reduce_lr])


# **Optimizer 3: Nadam**

model_nadam = Model(input_tensor, x)
model_nadam.compile(optimizer=Nadam(), loss='binary_crossentropy', metrics=['accuracy'])
hist_nadam = model_nadam.fit(X_train, y_train, batch_size=batch_size*2,
                epochs=300, validation_data=(X_valid,y_valid), verbose=0, callbacks=[reduce_lr])


# **Optimizer 4: SGD**

model_sgd = Model(input_tensor, x)
model_sgd.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
hist_sgd = model_sgd.fit(X_train, y_train, batch_size=batch_size*2,
              epochs=300, validation_data=(X_valid,y_valid), verbose=0, callbacks=[reduce_lr])


# **Optimizer 5: SGD + Nesterov**

model_sgdnes = Model(input_tensor, x)
model_sgdnes.compile(optimizer=SGD(nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
hist_sgdnes = model_sgdnes.fit(X_train, y_train, batch_size=batch_size*2,
              epochs=300, validation_data=(X_valid,y_valid), verbose=0, callbacks=[reduce_lr])


# **Optimizer 6: SGD with momentum=0.9**

model_sgdmo = Model(input_tensor, x)
model_sgdmo.compile(optimizer=SGD(momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
hist_sgdmo = model_sgdmo.fit(X_train, y_train, batch_size=batch_size*2,
              epochs=300, validation_data=(X_valid,y_valid), verbose=0, callbacks=[reduce_lr])


# **Optimizer 7: SGD + Nesterov with momentum=0.9 **

model_sgdmones = Model(input_tensor, x)
model_sgdmones.compile(optimizer=SGD(momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
hist_sgdmones = model_sgdmones.fit(X_train, y_train, batch_size=batch_size*2,
              epochs=300, validation_data=(X_valid,y_valid), verbose=0, callbacks=[reduce_lr])


# **Optimizer 8: ALECO2 with dP=0.01 and xi=0.85 **

model_aleco2 = Model(input_tensor, x)
model_aleco2.compile(optimizer=ALECO2(dP=0.01, xi=0.85), loss='binary_crossentropy', metrics=['accuracy'])
hist_sgdmones = model_sgdmones.fit(X_train, y_train, batch_size=batch_size*2,
              epochs=300, validation_data=(X_valid,y_valid), verbose=0)


# Save results in a .txt file

# # Validation Accuracy # #

numpy.savetxt("dogscats_results/val_acc_RMSprop.txt",numpy.array(hist_rmsprop.history['val_acc']), delimiter=",")
numpy.savetxt("dogscats_results/val_acc_Adam.txt",numpy.array(hist_adam.history['val_acc']), delimiter=",")
numpy.savetxt("dogscats_results/val_acc_Nadam.txt",numpy.array(hist_nadam.history['val_acc']), delimiter=",")
numpy.savetxt("dogscats_results/val_acc_SGD.txt",numpy.array(hist_sgd.history['val_acc']), delimiter=",")
numpy.savetxt("dogscats_results/val_acc_SGDNes.txt",numpy.array(hist_sgdnes.history['val_acc']), delimiter=",")
numpy.savetxt("dogscats_results/val_acc_SGDMo.txt",numpy.array(hist_sgdmo.history['val_acc']), delimiter=",")
numpy.savetxt("dogscats_results/val_acc_SGDMoNes.txt",numpy.array(hist_sgdmones.history['val_acc']), delimiter=",")
numpy.savetxt("dogscats_results/val_acc_ALECO2.txt",numpy.array(hist_aleco2.history['val_acc']), delimiter=",")


# # Accuracy # #

numpy.savetxt("dogscats_results/acc_RMSprop.txt",numpy.array(hist_rmsprop.history['acc']), delimiter=",")
numpy.savetxt("dogscats_results/acc_Adam.txt",numpy.array(hist_adam.history['acc']), delimiter=",")
numpy.savetxt("dogscats_results/acc_Nadam.txt",numpy.array(hist_nadam.history['acc']), delimiter=",")
numpy.savetxt("dogscats_results/acc_SGD.txt",numpy.array(hist_sgd.history['acc']), delimiter=",")
numpy.savetxt("dogscats_results/acc_SGDNes.txt",numpy.array(hist_sgdnes.history['acc']), delimiter=",")
numpy.savetxt("dogscats_results/acc_SGDMo.txt",numpy.array(hist_sgdmo.history['acc']), delimiter=",")
numpy.savetxt("dogscats_results/acc_SGDMoNes.txt",numpy.array(hist_sgdmones.history['acc']), delimiter=",")
numpy.savetxt("dogscats_results/acc_ALECO2.txt",numpy.array(hist_aleco2.history['acc']), delimiter=",")


# # Validation Loss # #

numpy.savetxt("dogscats_results/val_loss_RMSprop.txt",numpy.array(hist_rmsprop.history['val_loss']), delimiter=",")
numpy.savetxt("dogscats_results/val_loss_Adam.txt",numpy.array(hist_adam.history['val_loss']), delimiter=",")
numpy.savetxt("dogscats_results/val_loss_Nadam.txt",numpy.array(hist_nadam.history['val_loss']), delimiter=",")
numpy.savetxt("dogscats_results/val_loss_SGD.txt",numpy.array(hist_sgd.history['val_loss']), delimiter=",")
numpy.savetxt("dogscats_results/val_loss_SGDNes.txt",numpy.array(hist_sgdnes.history['val_loss']), delimiter=",")
numpy.savetxt("dogscats_results/val_loss_SGDMo.txt",numpy.array(hist_sgdmo.history['val_loss']), delimiter=",")
numpy.savetxt("dogscats_results/val_loss_SGDMoNes.txt",numpy.array(hist_sgdmones.history['val_loss']), delimiter=",")
numpy.savetxt("dogscats_results/val_loss_ALECO2.txt",numpy.array(hist_aleco2.history['val_loss']), delimiter=",")


# # Loss # #

numpy.savetxt("dogscats_results/loss_RMSprop.txt",numpy.array(hist_rmsprop.history['loss']), delimiter=",")
numpy.savetxt("dogscats_results/loss_Adam.txt",numpy.array(hist_adam.history['loss']), delimiter=",")
numpy.savetxt("dogscats_results/loss_Nadam.txt",numpy.array(hist_nadam.history['loss']), delimiter=",")
numpy.savetxt("dogscats_results/loss_SGD.txt",numpy.array(hist_sgd.history['loss']), delimiter=",")
numpy.savetxt("dogscats_results/loss_SGDNes.txt",numpy.array(hist_sgdnes.history['loss']), delimiter=",")
numpy.savetxt("dogscats_results/loss_SGDMo.txt",numpy.array(hist_sgdmo.history['loss']), delimiter=",")
numpy.savetxt("dogscats_results/loss_SGDMoNes.txt",numpy.array(hist_sgdmones.history['loss']), delimiter=",")
numpy.savetxt("dogscats_results/loss_ALECO2.txt",numpy.array(hist_aleco2.history['loss']), delimiter=",")
