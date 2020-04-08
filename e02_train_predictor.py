#!/usr/bin/python3
# -*- coding: utf-8 -*-
__version__ = '0.0.1' # Time-stamp: <2020-02-04T08:41:25Z>

"""Train Predictor."""

import time
TIME_INVOKED = time.time()
print(time.strftime("Invoked at %Y-%m-%d %H:%M:%S %Z",
                    time.localtime(TIME_INVOKED)))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
import gym
import matplotlib.pyplot as plt
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--save-params", default="predictor.hdf5", type=str)
parser.add_argument("--save-options", default="predictor.csv", type=str)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--steps", default=300, type=int)
parser.add_argument("--learning-rate", default=0.01, type=float)
parser.add_argument("--optimizer", default="rmsprop", choices=["sgd", "rmsprop", "adam"])
parser.add_argument("--batch-size", default=50, type=int)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--power", default=0.5, type=float) # default=0.0015
parser.add_argument("--max-speed", default=1.0, type=float) #default=0.07

ARGS = parser.parse_args()


def format_interval (interval):
    import math
    s = interval
    m = math.floor(s / 60)
    if m <= 0:
        return "{:f}s".format(s)
    s -= 60 * m
    r = "{:f}s".format(s)
    if s < 10:
        r = "0" + r
    h = math.floor(m / 60)
    if h <= 0:
        return "{}m".format(m) + r
    m -= 60 * h
    r = "{:02}m".format(m) + r
    d = math.floor(h / 24)
    if d <= 0:
        return "{}h".format(h) + r
    h -= 24 * d
    return "{}d{:02}h".format(d, h) + r

def save_options (path, options):
    import csv
    with open(path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Name', 'Value', 'Type'])
        writer.writeheader()
        r = []
        for k, v in options.items():
            if v is None:
                t = 'None'
            elif isinstance(v, bool):
                t = 'bool'
            elif isinstance(v, int):
                t = 'int'
            elif isinstance(v, float):
                t = 'float'
            else:
                t = 'str'
            r.append({'Name': k, 'Value': v, 'Type': t})
        writer.writerows(r)

def load_options (path):
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        r = {}
        for row in reader:
            v = row['Value']
            t = row['Type']
            if t == 'None':
                v = None
            elif t == 'bool':
                v = bool(v)
            elif t == 'int':
                v = int(v)
            elif t == 'float':
                v = float(v)
            r[row['Name']] = v
        return r


class CarEnv:
    def __init__ (self, power=None, max_speed=None, seed=None):
        self.env = gym.make("MountainCarContinuous-v0")
        if seed is None:
            self.seed = self.env.seed()[0]
        else:
            self.seed = self.env.seed(seed)[0]
        if power is not None:
            self.env.unwrapped.power = power
        if max_speed is not None:
            self.env.unwrapped.max_speed = max_speed

    def calc_next_state(self, pos, vel, act):
        env = self.env
        env.unwrapped.state = np.array([pos, vel])
        act = np.sign(act) * (act ** 2)
        state, reward, done, info = env.step([act])
        return state

    def generate_batch(self, batch_size=50):
        env = self.env
        env.reset()
        while True:
            pos = np.random.uniform(env.min_position, env.max_position,
                                    (batch_size, 1))
            vel = np.random.uniform(- env.max_speed, env.max_speed,
                                    (batch_size, 1))
            act = np.random.uniform(- 1.0, 1.0,
                                    (batch_size, 1))
            current = np.concatenate([pos, vel], axis=1)
            purpose = np.array(list([
                self.calc_next_state(p[0], v[0], a[0])
                for p, v, a in zip(pos, vel, act)
            ]))
            yield current, act, purpose

    def generate_predictor_batch (self, batch_size=50):
        for current, action, purpose \
            in self.generate_batch(batch_size=batch_size):
            yield (np.concatenate([current, action], axis=1), purpose)


class Predictor:
    def __init__ (self, learning_rate=0.001, optimizer="sgd"):
        self.num_states = 2
        self.num_actions = 1
        self.learning_rate = learning_rate
        self.model = self._build_network()
        if optimizer == 'rmsprop':
            opt = RMSprop(learning_rate)
        elif optimizer == 'adam':
            opt = Adam(learning_rate)
        else:
            opt = SGD(learning_rate)
        self.model.compile(optimizer=opt, loss='mse')

    def _build_network (self):
        model = Sequential([
            Dense(30, input_shape=(self.num_states + self.num_actions,)),
            Activation('sigmoid'),
            Dense(int(np.sqrt(30 * 10))),
            Activation('sigmoid'),
            Dense(self.num_states)
        ])
        model.summary()
        return model


def show (history):
#    acc = history.history['acc']
#    val_acc = history.history['val_acc']
    loss = history.history['loss']
#    val_loss = history.history['val_loss']

    epochs = range(len(loss))

#    plt.plot(epochs, acc, 'bo' ,label = 'training acc')
#    plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
#    plt.title('Training and Validation acc')
#    plt.legend()

#    plt.figure()

    plt.plot(epochs, loss, 'b' ,label = 'training loss')
#    plt.plot(epochs, loss, 'bo' ,label = 'training loss')
#    plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
#    plt.title('Training and Validation loss')
    plt.title('Training loss')
    plt.legend()

    plt.show()
    

def run ():
    cenv= CarEnv(power=ARGS.power, max_speed=ARGS.max_speed,
                 seed=ARGS.seed)
    pred = Predictor(learning_rate=ARGS.learning_rate)
    gen = cenv.generate_predictor_batch(batch_size=ARGS.batch_size)
    history = pred.model.fit(gen,
                             steps_per_epoch=ARGS.steps,
                             epochs=ARGS.epochs, verbose=1)
    score = pred.model.evaluate(gen,
                                steps=ARGS.steps,
                                verbose=1)
    print("Test Score: ", score)
    pred.model.save_weights(ARGS.save_params)

    save_options(ARGS.save_options, {'power': ARGS.power,
                                     'max_speed': ARGS.max_speed,
                                     'batch_size': ARGS.batch_size,
                                     'learning_rate': ARGS.learning_rate,
                                     'optimizer': ARGS.optimizer,
                                     'seed': cenv.seed})
    return history


if __name__ == "__main__":
    history = None
    try:
        history = run()
    except:
        traceback.print_exc()

    time_end = time.time()
    print("Execution Time: " + format_interval(time_end - TIME_INVOKED))
    if ARGS.epochs != 0 and history is not None:
        show(history)
