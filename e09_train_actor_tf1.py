#!/usr/bin/python3
# -*- coding: utf-8 -*-
__version__ = '0.0.1' # Time-stamp: <2020-02-04T15:31:19Z>

"""Train Actor."""

import time
TIME_INVOKED = time.time()
print(time.strftime("Invoked at %Y-%m-%d %H:%M:%S %Z",
                    time.localtime(TIME_INVOKED)))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
import gym
import matplotlib.pyplot as plt
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--save-params", default="actor.hdf5", type=str)
parser.add_argument("--save-options", default="actor.csv", type=str)
parser.add_argument("--predictor-params", default="predictor.hdf5", type=str)
parser.add_argument("--predictor-options", default="predictor.csv", type=str)
parser.add_argument("--epochs", default=20, dest="epochs", type=int)
parser.add_argument("--steps", default=300, dest="steps", type=int)
parser.add_argument("--learning-rate", default=0.01, type=float)
parser.add_argument("--batch-size", default=50, type=int)
parser.add_argument("--pseudo-action-iteration", default=50, type=int)
parser.add_argument("--grad-coeff", default=0.1, type=float)
parser.add_argument("--optimizer", default="adam", choices=["sgd", "rmsprop", "adam"])
parser.add_argument("--train-predictor", default=False, action="store_true")
parser.add_argument("--train-actual", default=False, action="store_true")
parser.add_argument("--train-true", default=False, action="store_true")
parser.add_argument("--seed", default=None, dest="seed", type=int)
parser.add_argument("--power", default=None, dest="power", type=float) # default=0.0015
parser.add_argument("--max-speed", default=None, type=float) #default=0.07

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

    def generate_actor_batch (self, batch_size=50):
        for current, action, purpose \
            in self.generate_batch(batch_size=batch_size):
            yield (np.concatenate([current, purpose], axis=1), action)


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


class Actor:
    def __init__ (self, cenv=None, predictor=None, learning_rate=0.001,
                  pseudo_action_iteration=10, grad_coeff=1.0,
                  optimizer='sgd', train_actual=False):
        self.num_states = 2
        self.num_actions = 1
        self.cenv = cenv
        self.train_actual = train_actual
        self.predictor = predictor
        self.pseudo_action_iteration = pseudo_action_iteration
        self.grad_coeff = grad_coeff
        self.learning_rate = learning_rate
        if optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.model = self._build_network()
        self._compile_graph(self.model)

    def _build_network (self):
        model = Sequential([
            Dense(60, input_shape=(self.num_states * 2,)),
            Activation('sigmoid'),
            Dense(int(np.sqrt(60 * 1))),
            Activation('sigmoid'),
            Dense(self.num_actions)
        ])
        model.summary()
        return model

    def _compile_graph (self, model):
        pred = self.predictor
        self.purpose = tf.placeholder(
            tf.float32, shape=(None, self.num_states))
        self.current = tf.placeholder(
            tf.float32, shape=(None, self.num_states))
        self.actual = tf.placeholder(
            tf.float32, shape=(None, self.num_states))
        self.pseudo_action = tf.placeholder(
            tf.float32, shape=(None, self.num_actions))
        self.action = tf.placeholder(
            tf.float32, shape=(None, self.num_actions))

        pout = pred.model(tf.concat([self.current, self.action], axis=1))
        grad = tf.gradients(K.mean(K.square(self.purpose - pout)), [self.action])
        self.temp_action = self.action \
            - self.grad_coeff \
            * tf.cast(tf.shape(self.action)[0], tf.float32) * grad[0]

        self.predicted = pout

        act = self.model(tf.concat([self.current, self.purpose], axis=1))
        self.loss = K.mean(K.square(self.pseudo_action - act))

        opt = self.optimizer
        self.minimize = opt.minimize(self.loss, var_list=self.model.trainable_weights)

    def calc_pseudo_action(self, sess, actual, current, purpose, action):
        feed_dict = {
            self.purpose: purpose,
            self.current: current,
            self.action: action
        }
        predicted = sess.run(self.predicted, feed_dict)
        feed_dict[self.purpose] = purpose - actual + predicted
        for i in range(self.pseudo_action_iteration):
            pseudo_action = sess.run(self.temp_action, feed_dict)
            feed_dict[self.action] = pseudo_action

        return pseudo_action
    
    def update (self, sess, current, purpose, pseudo_action):
        feed_dict = {
            self.purpose: purpose,
            self.current: current,
            self.pseudo_action: pseudo_action
        }
        _, loss = sess.run([self.minimize, self.loss],
                           feed_dict)

        return loss

    def train (self, sess, inputs):
        current = inputs[:, 0:self.num_states]
        purpose = inputs[:, self.num_states:]

        action = self.model.predict(inputs)
        action = np.clip(action, -1.0, 1.0)
        actual = np.array([
            self.cenv.calc_next_state(state[0], state[1], act[0])
                          for state, act in zip(current, action)
        ])

        pseudo_action = self.calc_pseudo_action \
            (sess, actual, current, purpose, action)
        pseudo_action = np.clip(pseudo_action, -1.0, 1.0)

        loss = self.update(sess, current, purpose, pseudo_action)
        if self.train_actual:
            loss2 = self.update(sess, current, actual, action)

        return loss, action, pseudo_action, actual


def show (history):
    loss = history['loss']
    tloss = history['tloss']
    ploss = history['ploss']

    epochs = range(len(loss))

    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(epochs, loss, label = 'pact vs act')
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(epochs, tloss, label = 'tact vs act')
    ax2.plot(epochs, ploss, label = 'tact vs pact')
    ax2.legend()
    
    plt.show()


def run (sess):
    sess.run(tf.global_variables_initializer())
    K.set_session(sess)

    pred_opt = load_options(ARGS.predictor_options)
    power = pred_opt['power']
    max_speed = pred_opt['max_speed']
    seed = pred_opt['seed']
    if ARGS.power is not None:
        power = ARGS.power
    if ARGS.max_speed is not None:
        max_speed = ARGS.max_speed
    if ARGS.seed is not None:
        seed = ARGS.seed
    cenv= CarEnv(power=power, max_speed=max_speed,
                 seed=seed)
    pred = Predictor(learning_rate=pred_opt['learning_rate'],
                     optimizer=pred_opt['optimizer'])
    pred.model.load_weights(ARGS.predictor_params)

    actor = Actor(cenv=cenv, predictor=pred,
                  learning_rate=ARGS.learning_rate,
                  pseudo_action_iteration=ARGS.pseudo_action_iteration,
                  grad_coeff=ARGS.grad_coeff,
                  optimizer=ARGS.optimizer,
                  train_actual=ARGS.train_actual)
    sess.run(tf.variables_initializer(actor.optimizer.variables()))
    gen = cenv.generate_actor_batch(batch_size=ARGS.batch_size)
    history = {'loss': [], 'ploss': [], 'tloss': []}

    for cur_epoch in range(ARGS.epochs):
        loss_sum = 0
        tloss_sum = 0
        ploss_sum = 0
        for cur_step in range(ARGS.steps):
            batch = gen.__next__()
            true_action = batch[1]
            current = batch[0][:, 0:actor.num_states]
            if ARGS.train_true:
                action = actor.model.predict(batch[0])
                pseudo_action = action
                actual = batch[0][:, actor.num_states:]
                loss = actor.update(sess, current, actual, true_action)
            else:
                loss, action, pseudo_action, actual \
                    = actor.train(sess, batch[0])
            tloss = np.mean(np.square(true_action - action))
            ploss = np.mean(np.square(true_action - pseudo_action))
            if ARGS.train_predictor:
                pred.model.train_on_batch \
                    (np.concatenate([current, action], axis=1), actual)
            loss_sum += loss
            tloss_sum += tloss
            ploss_sum += ploss
        loss = loss_sum / ARGS.steps
        tloss = tloss_sum / ARGS.steps
        ploss = ploss_sum / ARGS.steps
        history['loss'].append(loss)
        history['tloss'].append(tloss)
        history['ploss'].append(ploss)
        print("Epoch: ", cur_epoch, " Step Loss: ", loss,
              " tLoss: ", tloss,
              " pLoss: ", ploss)

    actor.model.save_weights(ARGS.save_params)

    save_options(ARGS.save_options, {
        'power': power,
        'max_speed': max_speed,
        'batch_size': ARGS.batch_size,
        'learning_rate': ARGS.learning_rate,
        'pseudo_action_iteration': ARGS.pseudo_action_iteration,
        'grad_coeff': ARGS.grad_coeff,
        'optimizer': ARGS.optimizer,
        'seed': cenv.seed
    })
    return history


if __name__ == "__main__":
    history = None
    try:
        with tf.Session() as sess:
            history = run(sess)
    except:
        traceback.print_exc()

    time_end = time.time()
    print("Execution Time: " + format_interval(time_end - TIME_INVOKED))
    if history is not None:
        show(history)
