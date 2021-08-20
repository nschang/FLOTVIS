# -*- coding: utf-8 -*-
import os
import math
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import scipy.signal

class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))  
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

class ExponentialDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 decay_rate,
                 verbose=0):
        super(ExponentialDecayScheduler, self).__init__()
        self.decay_rate         = decay_rate
        self.verbose            = verbose
        self.learning_rates     = []

    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))

class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self, T_max, eta_min=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.T_max      = T_max
        self.eta_min    = eta_min
        self.verbose    = verbose
        self.init_lr    = 0
        self.last_epoch = 0

    def on_train_begin(self, batch, logs=None):
        self.init_lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, batch, logs=None):
        learning_rate = self.eta_min + (self.init_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        self.last_epoch += 1

        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))

# ------------------------------
# compute callbacks
# ------------------------------
# def cosine_decay_with_warmup(global_step,
#                              learning_rate_base,
#                              total_steps,
#                              warmup_learning_rate=0.0,
#                              warmup_steps=0,
#                              hold_base_rate_steps=0,
#                              min_learn_rate=0,
#                              ):
#     """
#     parameters:
#             global_step: Tcur, write current step
#             learning_rate_base：set base learning rate, learning rate decreases once this is reached during warmup
#             total_steps: total training steps = epoch * sample_count / batch_size
#             warmup_learning_rate: initial learning rate during warm up
#             warmup_steps: total steps for warmup warm_up
#             hold_base_rate_steps: number of steps to keep learning rate steady after warmup (optional)
#     """
#     if total_steps < warmup_steps:
#         raise ValueError('total_steps must be larger or equal to '
#                             'warmup_steps.')
#     # here we use cosine annealing and min learning rate is 0 
#     learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
#         (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
#     # when hold_base_rate_steps > 0, learning rate is steady for this many steps after warm up
#     if hold_base_rate_steps > 0:
#         learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
#                                     learning_rate, learning_rate_base)
#     if warmup_steps > 0:
#         if learning_rate_base < warmup_learning_rate:
#             raise ValueError('learning_rate_base must be larger or equal to '
#                                 'warmup_learning_rate.')
#         # linear growth
#         slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
#         warmup_rate = slope * global_step + warmup_learning_rate
#         # use linear warmup_rate only during warm up of global_step
#         # otherwise use learning_rate of cosine annealingf
#         learning_rate = np.where(global_step < warmup_steps, warmup_rate,
#                                     learning_rate)

#     learning_rate = max(learning_rate,min_learn_rate)
#     return learning_rate

# class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
#     """
#     learning rate scheduler using Callback
#     """
#     def __init__(self,
#                  learning_rate_base,
#                  total_steps,
#                  global_step_init=0,
#                  warmup_learning_rate=0.0,
#                  warmup_steps=0,
#                  hold_base_rate_steps=0,
#                  min_learn_rate=0,
#                  # interval_epoch = lowest value between cosine annealing
#                  interval_epoch=[0.05, 0.15, 0.30, 0.50],
#                  verbose=0):
#         super(WarmUpCosineDecayScheduler, self).__init__()
#         # base learning rate
#         self.learning_rate_base = learning_rate_base
#         # adjust parameter
#         self.warmup_learning_rate = warmup_learning_rate
#         # show parameter
#         self.verbose = verbose
#         # learning_rates after each update to visualize result
#         self.min_learn_rate = min_learn_rate
#         self.learning_rates = []

#         self.interval_epoch = interval_epoch
#         # global step
#         self.global_step_for_interval = global_step_init
#         # total step number for increasing learning rate during warmup
#         self.warmup_steps_for_interval = warmup_steps
#         # use total step number of peak
#         self.hold_steps_for_interval = hold_base_rate_steps
#         # total steps of training
#         self.total_steps_for_interval = total_steps

#         self.interval_index = 0
#         # compute interval between two lows
#         self.interval_reset = [self.interval_epoch[0]]
#         for i in range(len(self.interval_epoch)-1):
#             self.interval_reset.append(self.interval_epoch[i+1]-self.interval_epoch[i])
#         self.interval_reset.append(1-self.interval_epoch[-1])

# 	# update global_step and write current learning rate
#     def on_batch_end(self, batch, logs=None):
#         self.global_step = self.global_step + 1
#         self.global_step_for_interval = self.global_step_for_interval + 1
#         lr = K.get_value(self.model.optimizer.lr)
#         self.learning_rates.append(lr)

# 	#update learning rate
#     def on_batch_begin(self, batch, logs=None):
#         # update parameters at each new low
#         if self.global_step_for_interval in [0]+[int(i*self.total_steps_for_interval) for i in self.interval_epoch]:
#             self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
#             self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
#             self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
#             self.global_step = 0
#             self.interval_index += 1

#         lr = cosine_decay_with_warmup(global_step=self.global_step,
#                                       learning_rate_base=self.learning_rate_base,
#                                       total_steps=self.total_steps,
#                                       warmup_learning_rate=self.warmup_learning_rate,
#                                       warmup_steps=self.warmup_steps,
#                                       hold_base_rate_steps=self.hold_base_rate_steps,
#                                       min_learn_rate = self.min_learn_rate)
#         K.set_value(self.model.optimizer.lr, lr)
#         if self.verbose > 0:
#             print('\nBatch %05d: setting learning '
#                   'rate to %s.' % (self.global_step + 1, lr))
