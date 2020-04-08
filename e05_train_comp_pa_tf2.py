#!/usr/bin/python3
# -*- coding: utf-8 -*-
__version__ = '0.0.1' # Time-stamp: <2020-02-04T14:40:56Z>

"""Train Actor and Predictor with Competitive Learning."""

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
import scipy.optimize
import matplotlib.pyplot as plt
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=50, dest="epochs", type=int)
parser.add_argument("--steps", default=100, dest="steps", type=int)
parser.add_argument("--batch-size", default=50, type=int)
parser.add_argument("--predictor-params", default="predictor.hdf5", type=str)
parser.add_argument("--predictor-options", default="predictor.csv", type=str)
parser.add_argument("--actor-params", default="actor.hdf5", type=str)
parser.add_argument("--actor-options", default="actor.csv", type=str)
parser.add_argument("--predictor-lr", default=0.01, type=float)
parser.add_argument("--predictor-comp-lr", default=0.005, type=float)
parser.add_argument("--predictor-optimizer", default="rmsprop", choices=["sgd", "rmsprop", "adam"])
parser.add_argument("--actor-lr", default=0.01, type=float)
parser.add_argument("--actor-comp-lr", default=0.05, type=float)
parser.add_argument("--actor-optimizer", default="adam", choices=["sgd", "rmsprop", "adam"])
parser.add_argument("--pseudo-action-iteration", default=50, type=int)
parser.add_argument("--grad-coeff", default=0.1, type=float)
parser.add_argument("--pre-train", default=0, type=int)
parser.add_argument("--debug-print", default=0, type=int)
parser.add_argument("--save-history", default=None, type=str)
parser.add_argument("--random-action", default=False, action="store_true")
parser.add_argument("--seed", default=None, dest="seed", type=int)
parser.add_argument("--power", default=0.5, dest="power", type=float) # default=0.0015
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

def save_history (path, history):
    import csv
    with open(path, 'w') as f:
        keys = list(history.keys())
        epochs = len(history[keys[0]])
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        r = [dict([[k, history[k][i]] for k in keys])
             for i in range(epochs)]
        writer.writerows(r)
    
def load_history (path):
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        h = {}
        for k in reader.fieldnames:
            h[k] = []
        for row in reader:
            for k in reader.fieldnames:
                h[k].append(float(row[k]))
        return h


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
    def __init__ (self, predictor=None, learning_rate=0.001,
                  pseudo_action_iteration=10, grad_coeff=1.0,
                  optimizer='sgd'):
        self.num_states = 2
        self.num_actions = 1
        self.predictor = predictor
        self.pseudo_action_iteration = pseudo_action_iteration
        self.grad_coeff = grad_coeff
        if optimizer == 'rmsprop':
            opt = RMSprop(learning_rate)
        elif optimizer == 'adam':
            opt = Adam(learning_rate)
        else:
            opt = SGD(learning_rate)
        self.model = self._build_network()
        self.model.compile(optimizer=opt,
                           loss='mse')

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

    @tf.function
    def calc_temp_action(self, current, purpose, action):
        pout = self.predictor.model \
            (tf.concat([current, action], axis=1))
        grad = tf.gradients(K.mean(K.square(purpose - pout)),
                            [action])
        pact = action \
            - self.grad_coeff \
            * tf.cast(tf.shape(action)[0], tf.float32) * grad[0]

        return pact, pout


def comp_train (cenv, pred, actor, current, purpose):
    action = actor.model(np.concatenate([current, purpose], axis=1)).numpy()
    action = np.clip(action, -1.0, 1.0)
    if ARGS.random_action:
        taction = action
        action = np.random.uniform(-1.0, 1.0, (current.shape[0], 1))

    pactions = []
    predicteds = []
    purpose = purpose.astype(np.float32)
    current = current.astype(np.float32)
    action = action.astype(np.float32)
    pseudo_action = action
    pactions.append(action)
    for i in range(actor.pseudo_action_iteration):
        pseudo_action, predicted = actor.calc_temp_action \
            (current, purpose, pseudo_action)
        pseudo_action = pseudo_action.numpy()
        predicted = predicted.numpy()
        pactions.append(pseudo_action)
        predicteds.append(predicted)
    pactions.pop()

    pseudo_action = np.clip(pseudo_action, -1.0, 1.0)
    pactual = np.array(list([
        cenv.calc_next_state(state[0], state[1], act[0])
        for state, act in zip(current, pseudo_action)
    ]))

    pred.model.optimizer.lr = ARGS.predictor_lr
    pred_loss = pred.model.train_on_batch(
        np.concatenate([current, pseudo_action], axis=1),
        pactual
    )
    actor.model.optimizer.lr = ARGS.actor_lr
    actor_loss = actor.model.train_on_batch(
        np.concatenate([current, pactual], axis=1),
        pseudo_action
    )

    pinputs = []
    poutputs = []
    delta = 0.5 * (pactual + purpose) - predicted
#    delta = pactual - predicted
    for i, (pa, pr) in enumerate(zip(pactions, predicteds)):
        y = delta * ((i + 1) / len(pactions)) + pr
        x = np.concatenate([current, pa], axis=1)
        pinputs.append(x)
        poutputs.append(y)
    pinputs = np.concatenate(pinputs, axis=0)
    poutputs = np.concatenate(poutputs, axis=0)
    pred.model.optimizer.lr = ARGS.predictor_comp_lr
    pred_comp_loss = pred.model.train_on_batch(pinputs, poutputs)

    negact = action - (pseudo_action - action)\
        * np.exp(- ((pseudo_action - action) / 2.0) ** 2) \
        * np.tanh(np.mean(((pactual - purpose) /
                           np.array([[cenv.env.power * 2,
                                      cenv.env.max_speed * 2]])) ** 2,
                          axis=1, keepdims=True))
    actor.model.optimizer.lr = ARGS.actor_comp_lr
    actor_comp_loss = actor.model.train_on_batch(
        np.concatenate([current, purpose], axis=1),
        negact
    )

    if ARGS.random_action:
        action = taction

    return action, pseudo_action, pactual, predicted, \
        pred_loss, actor_loss, \
        pred_comp_loss, actor_comp_loss
    

def show (history):
    taloss = history['tloss']
    paloss = history['ploss']
    aloss = history['actor_loss']
    ploss = history['pred_loss']
    acloss = history['actor_comp_loss']
    pcloss = history['pred_comp_loss']

    epochs = range(len(taloss))

    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(epochs, aloss, label = 'aLoss')
    ax1.plot(epochs, ploss, label = 'pLoss')
    ax1.plot(epochs, acloss, label = 'acLoss')
    ax1.plot(epochs, pcloss, label = 'pcLoss')
#    plt.plot(epochs, loss, 'bo' ,label = 'training loss')
#    plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(epochs, taloss, label = 'tact vs act')
    ax2.plot(epochs, paloss, label = 'tact vs pact')
    ax2.legend()
    
    plt.show()

    
def run ():
    power = ARGS.power
    max_speed = ARGS.max_speed
    seed = ARGS.seed
    
    cenv= CarEnv(power=power, max_speed=max_speed,
                 seed=seed)
    pred = Predictor(learning_rate=ARGS.predictor_lr,
                     optimizer=ARGS.predictor_optimizer)
    actor = Actor(predictor=pred,
                  learning_rate=ARGS.actor_lr,
                  pseudo_action_iteration=ARGS.pseudo_action_iteration,
                  grad_coeff=ARGS.grad_coeff,
                  optimizer=ARGS.actor_optimizer)
    gen = cenv.generate_batch(batch_size=ARGS.batch_size)

    history = {'ploss': [], 'tloss': [],
               'aploss': [], 'pploss': [],
               'actor_loss': [], 'pred_loss': [],
               'actor_comp_loss': [], 'pred_comp_loss': []}

    for i in range(ARGS.pre_train):
        current, action, purpose = gen.__next__()
        pred.model.train_on_batch(np.concatenate([current, action], axis=1),
                                  purpose)
        actor.model.train_on_batch(np.concatenate([current, purpose], axis=1),
                                   action)

    for cur_epoch in range(ARGS.epochs):
        tloss_sum = 0
        ploss_sum = 0
        aploss_sum = 0
        pploss_sum = 0
        actor_loss_sum = 0
        pred_loss_sum = 0
        actor_comp_loss_sum = 0
        pred_comp_loss_sum = 0
        for cur_step in range(ARGS.steps):
            current, true_action, purpose = gen.__next__()
            action, pseudo_action, pactual, predicted, \
                pred_loss, actor_loss, \
                pred_comp_loss, actor_comp_loss \
                = comp_train(cenv, pred, actor, current, purpose)
            tloss = np.mean(np.square(true_action - action))
            ploss = np.mean(np.square(true_action - pseudo_action))
            aploss = np.mean(np.square(pactual - purpose))
            pploss = np.mean(np.square(purpose - predicted))
            tloss_sum += tloss
            ploss_sum += ploss
            aploss_sum += aploss
            pploss_sum += pploss
            actor_loss_sum += actor_loss
            pred_loss_sum += pred_loss
            actor_comp_loss_sum += actor_comp_loss
            pred_comp_loss_sum += pred_comp_loss
        if ARGS.debug_print > 0:
            print("current:", current,
                  "purpose:", purpose,
                  "pactual:", pactual,
                  "true_action:", true_action,
                  "pseudo_action:", pseudo_action,
                  "action:", action)
        tloss = tloss_sum / ARGS.steps
        ploss = ploss_sum / ARGS.steps
        aploss = aploss_sum / ARGS.steps
        pploss = pploss_sum / ARGS.steps
        actor_loss = actor_loss_sum / ARGS.steps
        pred_loss = pred_loss_sum / ARGS.steps
        actor_comp_loss = actor_comp_loss_sum / ARGS.steps
        pred_comp_loss = pred_comp_loss_sum / ARGS.steps
        history['tloss'].append(tloss)
        history['ploss'].append(ploss)
        history['aploss'].append(aploss)
        history['pploss'].append(pploss)
        history['actor_loss'].append(actor_loss)
        history['pred_loss'].append(pred_loss)
        history['actor_comp_loss'].append(actor_comp_loss)
        history['pred_comp_loss'].append(pred_comp_loss)
        print("Epoch:", cur_epoch, " Step", 
              "taLoss:", tloss,
              "paLoss:", ploss,
              "apLoss:", aploss,
              "ppLoss:", pploss,
              "aLoss:", actor_loss,
              "pLoss:", pred_loss,
              "acLoss:", actor_comp_loss,
              "pcLoss:", pred_comp_loss
        )

    pred.model.save_weights(ARGS.predictor_params)
    save_options(ARGS.predictor_options, {
        'power': power,
        'max_speed': max_speed,
        'batch_size': ARGS.batch_size,
        'learning_rate': ARGS.predictor_lr,
        'comp_learning_rate': ARGS.predictor_comp_lr,
        'optimizer': ARGS.predictor_optimizer,
        'seed': cenv.seed
    })

    actor.model.save_weights(ARGS.actor_params)
    save_options(ARGS.actor_options, {
        'power': power,
        'max_speed': max_speed,
        'batch_size': ARGS.batch_size,
        'learning_rate': ARGS.actor_lr,
        'comp_learning_rate': ARGS.actor_comp_lr,
        'pseudo_action_iteration': ARGS.pseudo_action_iteration,
        'grad_coeff': ARGS.grad_coeff,
        'optimizer': ARGS.actor_optimizer,
        'seed': cenv.seed
    })

    if ARGS.save_history is not None:
        save_history(ARGS.save_history, history)

    return history


if __name__ == "__main__":
    history = None
    try:
        history = run()
    except:
        traceback.print_exc()

    time_end = time.time()
    print("Execution Time: " + format_interval(time_end - TIME_INVOKED))
    if history is not None:
        show(history)
