#!/usr/bin/python3
# -*- coding: utf-8 -*-
__version__ = '0.0.1' # Time-stamp: <2020-02-02T12:15:51Z>

"""Test by Recollector."""

import time
TIME_INVOKED = time.time()
print(time.strftime("Invoked at %Y-%m-%d %H:%M:%S %Z",
                    time.localtime(TIME_INVOKED)))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import gym
import scipy.optimize
import matplotlib.pyplot as plt
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--predictor-params", default="predictor.hdf5", type=str)
parser.add_argument("--predictor-options", default="predictor.csv", type=str)
parser.add_argument("--actor-params", default="actor.hdf5", type=str)
parser.add_argument("--actor-options", default="actor.csv", type=str)
parser.add_argument("--length", default=5, dest="length", type=int)
parser.add_argument("--method", default="da", choices=["de", "shgo", "da", "da0"])
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--power", default=None,type=float) # default=0.0015
parser.add_argument("--max-speed", default=None, type=float) #default=0.07
parser.add_argument("--sleep", default=1, type=float)
parser.add_argument("--reset-sleep", default=0, type=float)
parser.add_argument("--end-sleep", default=0, type=float)
parser.add_argument("--velocity-reward", default="last", choices=["none", "sum", "last", "neg"])

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
    # def __init__ (self, cenv=None, predictor=None, learning_rate=0.001,
    #               pseudo_action_iteration=10, grad_coeff=1.0,
    #               optimizer='sgd'):
    def __init__ (self, predictor=None, learning_rate=0.001,
                  pseudo_action_iteration=10, grad_coeff=1.0,
                  optimizer='sgd'):
        self.num_states = 2
        self.num_actions = 1
        # self.cenv = cenv
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
        # inputs = Input(shape = (self.num_states * 2,))
        # dense1 = Dense(60, activation='sigmoid')(inputs)
        # dense2 = Dense(int(np.sqrt(60 * 1)), activation='sigmoid')(dense1)
        # outputs = Dense(self.num_actions)(dense2)
        # model = Model(inputs=[inputs], outputs=[outputs])
        # model.summary()
        # return model

    def _compile_graph (self, model):
        pred = self.predictor
        self.purpose = tf.placeholder(
            tf.float32, shape=(None, self.num_states))
        # self.actual = tf.placeholder(
        #     tf.float32, shape=(None, self.num_states))
        self.current = tf.placeholder(
            tf.float32, shape=(None, self.num_states))
        self.pseudo_action = tf.placeholder(
            tf.float32, shape=(None, self.num_actions))
        self.action = tf.placeholder(
            tf.float32, shape=(None, self.num_actions))

        pout = pred.model(tf.concat([self.current, self.action], axis=1))
        # y = tf.stop_gradient(self.purpose - self.actual + pout)
        # grad = tf.gradients(K.mean(K.square(y - pout)), [self.action])
        grad = tf.gradients(K.mean(K.square(self.purpose - pout)), [self.action])
        self.temp_action = self.action \
            - self.grad_coeff \
            * tf.cast(tf.shape(self.action)[0], tf.float32) * grad[0]

        act = self.model(tf.concat([self.current, self.purpose], axis=1))
        self.loss = K.mean(K.square(self.pseudo_action - act))

        opt = self.optimizer
        self.minimize = opt.minimize(self.loss, var_list=self.model.trainable_weights)

    # def calc_pseudo_action(self, sess, actual, current, purpose, action):
    def calc_pseudo_action(self, sess, current, purpose, action):
        feed_dict = {
            self.purpose: purpose,
            # self.actual: actual,
            self.current: current,
            self.action: action
        }
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
        # actual = np.array([
        #     self.cenv.calc_next_state(state[0], state[1], act[0])
        #                   for state, act in zip(current, action)
        # ])

        # pseudo_action = self.calc_pseudo_action \
        #     (sess, actual, current, purpose, action)
        pseudo_action = self.calc_pseudo_action \
            (sess, current, purpose, action)
        pseudo_action = np.clip(pseudo_action, -1.0, 1.0)

        loss = self.update(sess, current, purpose, pseudo_action)
        return loss, action, pseudo_action


class Recollector:
    def __init__ (self, power=None, max_speed=None,
                  velocity_reward="none", seed=None,
                  length=5, method="da"):
        self.env = gym.make("MountainCarContinuous-v0")
        if seed is None:
            self.seed = self.env.seed()[0]
        else:
            self.seed = self.env.seed(seed)[0]
        self.velocity_reward = velocity_reward
        if power is not None:
            self.env.unwrapped.power = power
        if max_speed is not None:
            self.env.unwrapped.max_speed = max_speed
        self.method = method
        self.length = length
        self.prev = []

    def _opt_main (self, l):
        self.env.seed(self.seed)
        observation = self.env.reset()
        sum_velocity = 0
        l = self.prev + list(l)
        for i, a in enumerate(l):
            observation, reward, done, info = self.env.step([a])
            position = observation[0]
            velocity = observation[1]
            sum_velocity += np.abs(velocity)
            if done:
                done = i + 1
                break
        if done:
            for i in range(done, len(l)):
                reward -= (l[i] ** 2) * 0.1
            if self.velocity_reward == "sum":
                reward = reward * 100 + sum_velocity * 10000
            elif self.velocity_reward == "last":
                reward = reward * 100 + np.abs(velocity) * 10000
            elif self.velocity_reward == "neg":
                reward = reward * 100 - np.abs(velocity) * 100
            else:
                reward -= 10 * (done - 1) / len(l)
                reward *= 100
        else:
            reward = min(sum_velocity, 100)
        return - reward

    def optimize (self):
        length = self.length - len(self.prev)
        if length <= 0:
            return None
        method = self.method

        if method == "de":
            res = scipy.optimize.differential_evolution \
                (self._opt_main, [(-1.0, 1.0)] * length)
#        elif method == "bh":
#            res = scipy.optimize.basinhopping \
#                (self._opt_main, [0.0] * length)
        elif method == "shgo":
            res = scipy.optimize.shgo \
                (self._opt_main, [(-1.0, 1.0)] * length)
        elif method == "da":
            res = scipy.optimize.dual_annealing \
                (self._opt_main, [(-1.0, 1.0)] * length)
        elif method == "da0":
            res = scipy.optimize.dual_annealing \
                (self._opt_main, [(-1.0, 1.0)] * length,
                 x0=([0.0] * length))
        else:
            raise ValueError("Illegal method: " + method)

        return res

    def current_state(self, prev):
        prev = [np.sign(a) * (a ** 2) for a in prev]
        self.env.seed(self.seed)
        observation = self.env.reset()
        for a in self.prev:
            observation, reward, done, info = self.env.step([a])
        return observation

    def done(self, prev):
        prev = [np.sign(a) * (a ** 2) for a in prev]
        self.env.seed(self.seed)
        observation = self.env.reset()
        for a in self.prev:
            observation, reward, done, info = self.env.step([a])
            if done:
                return True
        return False

    def purpose_state(self, prev):
        self.prev = [np.sign(a) * (a ** 2) for a in prev]
        res = self.optimize()
        if res is None:
            raise ValueError("Already Ended.")
        reward = - self._opt_main(res.x)
        if not hasattr(res, 'success') or res.success:
            print("OptForStep: iterated {0} times score={1}"
                  .format(res.nit, reward))
        else:
            print("OptForStep(Fail): iterated {0} times score={1}"
                  .format(res.nit, reward))
        print(res.x)
        self.prev.append(res.x[0])
        self.env.seed(self.seed)
        self.env.reset()
        for a in self.prev:
            observation, reward, done, info = self.env.step([a])

        return observation


def show (recoll, history):
    env = recoll.env
    env.seed(recoll.seed)
    env.reset()
    env.render()
    time.sleep(ARGS.sleep)
    time.sleep(ARGS.reset_sleep)
    for a in history:
        env.step([np.sign(a) * (a ** 2)])
        env.render()
        time.sleep(ARGS.sleep)
    time.sleep(ARGS.end_sleep)

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

    act_opt = load_options(ARGS.actor_options)
    # actor = Actor(cenv=cenv, predictor=pred,
    actor = Actor(predictor=pred,
                  learning_rate=act_opt['learning_rate'],
                  pseudo_action_iteration=act_opt['pseudo_action_iteration'],
                  grad_coeff=act_opt['grad_coeff'],
                  optimizer=act_opt['optimizer'])
    sess.run(tf.variables_initializer(actor.optimizer.variables()))
    actor.model.load_weights(ARGS.actor_params)

    length = ARGS.length
    recoll = Recollector(
        power=power, max_speed=max_speed,
        velocity_reward=ARGS.velocity_reward, seed=ARGS.seed,
        length=ARGS.length, method=ARGS.method
    )

    history = []
    for i in range(length):
        current = np.array(recoll.current_state(history))
        if recoll.done(history):
            break
        purpose = np.array(recoll.purpose_state(history))
        act = actor.model.predict(
            np.concatenate([current, purpose]).reshape(1,4)
        )[0, 0]
        print("current: ", current, " purpose: ", purpose,
              " act: ", act, " act^2: ", np.sign(act) * (act ** 2))
        history.append(act)

    if recoll.done(history):
        print("Done: ", history)
    else:
        print("Fail: ", history)

    return recoll, history


if __name__ == "__main__":
    history = None
    try:
        with tf.Session() as sess:
            recoll, history = run(sess)
    except:
        traceback.print_exc()

    time_end = time.time()
    print("Execution Time: " + format_interval(time_end - TIME_INVOKED))
    if history is not None:
        show(recoll, history)
