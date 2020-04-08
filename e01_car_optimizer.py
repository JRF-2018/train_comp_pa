#!/usr/bin/python3
# -*- coding: utf-8 -*-
__version__ = '0.0.1' # Time-stamp: <2020-02-02T17:16:11Z>

"""A simple simulation of MountainCarContinuous."""

import time
TIME_INVOKED = time.time()
print(time.strftime("Invoked at %Y-%m-%d %H:%M:%S %Z",
                    time.localtime(TIME_INVOKED)))
import numpy as np
import matplotlib.pyplot as plt
import gym
import scipy.optimize
import traceback
import argparse

ALARM_FILE = "c:/Windows/media/Alarm01.wav"
import os
if not os.path.exists(ALARM_FILE):
    ALARM_FILE = None

parser = argparse.ArgumentParser()
parser.add_argument("--length", default=150, type=int)
parser.add_argument("--method", default="da", choices=["de", "shgo", "da", "da0"])
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--power", default=None, type=float) # default=0.0015
parser.add_argument("--max-speed", default=None, type=float) #default=0.07
parser.add_argument("--sleep", default=0.1, type=float)
parser.add_argument("--reset-sleep", default=0, type=float)
parser.add_argument("--end-sleep", default=0, type=float)
parser.add_argument("--velocity-reward", default="none", choices=["none", "sum", "last", "neg"])
parser.add_argument("--alarm", default=None, type=str)

ARGS = parser.parse_args()

if ARGS.alarm is not None:
    if ARGS.alarm == "":
        ALARM_FILE = None
    else:
        ALARM_FILE = ARGS.alarm

def alarm (file):
    import time
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  
    import pygame.mixer
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play(1)
    while (pygame.mixer.music.get_busy()):
        time.sleep(0.5)
    pygame.mixer.music.stop()


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


class CarOptimizer:
    def __init__ (self, power=None, max_speed=None,
                  velocity_reward="none", seed=None):
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

    def _opt_main (self, l):
        self.env.seed(self.seed)
        observation = self.env.reset()
        sum_velocity = 0
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

    def optimize (self, seed=None, method="bh", length=4):
        if seed is None:
            self.seed = self.env.seed()[0]
        else:
            self.seed = seed

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


def run():
    copt = CarOptimizer(power=ARGS.power, max_speed=ARGS.max_speed,
                        velocity_reward=ARGS.velocity_reward,
                        seed=ARGS.seed)
    env = copt.env
    seed = copt.seed
        
    res = copt.optimize(method=ARGS.method, length=ARGS.length)
    reward = - copt._opt_main(res.x)
    if not hasattr(res, 'success') or res.success:
        print("OptForStep: iterated {0} times score={1}"
              .format(res.nit, reward))
    else:
        print("OptForStep(Fail): iterated {0} times score={1}"
              .format(res.nit, reward))
    print(res.x)

    if ALARM_FILE is not None:
        alarm(ALARM_FILE)

    env.seed(seed)
    env.reset()
    env.render()
    time.sleep(ARGS.sleep)
    time.sleep(ARGS.reset_sleep)
    for i, a in enumerate(res.x):
        observation, reward, done, info = env.step([a])
        env.render()
        time.sleep(ARGS.sleep)
        if done:
            print("Done at ", i)
            break
    time.sleep(ARGS.end_sleep)


if __name__ == "__main__":
    try:
        run()
    except:
        traceback.print_exc()

    time_end = time.time()
    print("Execution Time: " + format_interval(time_end - TIME_INVOKED))
