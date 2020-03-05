import datetime
import math
import os

import numpy as np
from termcolor import colored


def perc_print(current, total, msg="", inverted=False):
    if msg:
        msg += ": "

    to_print = current / total

    if inverted:
        to_print = 1 - to_print

    to_print = round(to_print * 100, 2)
    print(colored(f"\r{msg}{to_print}% ", "yellow"), end="")

    return round(to_print)


def time_print(time_in_seconds):
    return str(datetime.timedelta(seconds=time_in_seconds))


def play_soud(duration=1, freq=500, repeat=1):
    """
    Play a sound
    :param duration: duration in seconds
    :param freq: frequence of the sound
    :return:
    """

    for i in range(repeat):
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))


class DebuggerPrint:

    def __init__(self, data_len, eval_perc, log_perc, num_epochs, batch_size):
        """
        :param data_lenght: (int) lenght of data list
        :param log_perc: (float) percentage of data lenght for logging
        :param eval_perc: (float) percentage of data lenght for evaluation
        :param num_epochs: (int) number of total epochs
        :param batch_size:  (int) batch size

        """

        self.data_len = data_len
        self.eval_perc = eval_perc
        self.log_perc = log_perc
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def perc_condition(self, current_idx, perc):
        real_idx = current_idx / self.batch_size
        condition = math.ceil(self.data_len * perc / self.batch_size)
        return real_idx % condition == 0

    def execution_infos(self, current_idx, execution_time, current_epoch):
        """
        Print infos bout ETA

        :param current_idx: (int) index of batch
        :param execution_time: (float) time since start of execution
        :param current_epoch: (int) current epoch
        :return:
        """

        current_idx = self.zero_division(current_idx)
        current_epoch = self.zero_division(current_epoch)

        log_steps = np.ceil(self.log_perc * self.data_len)
        eval_step = np.ceil(self.eval_perc * self.data_len)
        real_index = current_idx // self.batch_size

        real_index = self.zero_division(real_index)

        time_per_step = execution_time / real_index / current_epoch
        epoch_eta = time_per_step * self.data_len // self.batch_size
        execution_eta = epoch_eta * (self.num_epochs - current_epoch)

        to_print = f"\nTotal number of steps per epoch : {self.data_len}\n"
        to_print += f"Steps for logging {log_steps}\n"
        to_print += f"Steps for evaluation: {eval_step}\n"
        to_print += f"Average time per step : {time_print(time_per_step)}\n"
        to_print += f"ETA per epoch : {time_print(epoch_eta)}\n"
        to_print += f"ETA per whole execution : {time_print(execution_eta)}\n"

        print(colored(to_print, color="cyan"))

    def epoch_info(self, loss, accuracy, current_idx, current_epoch, execution_time):
        current_idx = self.zero_division(current_idx)
        current_epoch = self.zero_division(current_epoch)

        real_index = current_idx // self.batch_size
        real_index = self.zero_division(real_index)

        average_loss = loss / real_index
        time_per_step = execution_time / real_index / self.batch_size / current_epoch
        eta_current_epoch = (self.data_len - current_idx) / self.batch_size * time_per_step

        to_print = f"\nAverage loss : {average_loss}\n"
        to_print += f"Accuracy : {accuracy}\n"
        to_print += f"Average time per step : {time_print(time_per_step)}\n"
        to_print += f"ETA per current epoch : {time_print(eta_current_epoch)}\n"

        print(colored(to_print, color="cyan"))

    def zero_division(self, number):
        if number == 0:
            return 1
        return number
