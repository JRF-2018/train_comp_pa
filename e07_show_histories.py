#!/usr/bin/python3
# -*- coding: utf-8 -*-
__version__ = '0.0.1' # Time-stamp: <2020-02-02T10:41:49Z>

"""Show histories."""

import time
TIME_INVOKED = time.time()
print(time.strftime("Invoked at %Y-%m-%d %H:%M:%S %Z",
                    time.localtime(TIME_INVOKED)))

import matplotlib.pyplot as plt
import argparse
import traceback


parser = argparse.ArgumentParser()
parser.add_argument("specs", metavar="LABSL=CSV", nargs='+', type=str)
parser.add_argument("--type", default="tloss", type=str)
parser.add_argument("--title", default="Loss about True Action vs Predicted Action", type=str)

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


def run():
    specs = []
    epochs = []
    for s in ARGS.specs:
        label, fname = s.split('=', 1)
        history = load_history(fname)
        epochs.append(len(history[ARGS.type]))
        specs.append({'label': label, 'history': history[ARGS.type]})

    epochs = range(max(epochs))
    for s in specs:
        plt.plot(epochs, s['history'], label=s['label'])
    plt.legend()
    if ARGS.title is not None:
        plt.title(ARGS.title)
    plt.show()

if __name__ == "__main__":
    try:
        run()
    except:
        traceback.print_exc()

    time_end = time.time()
    print("Execution Time: " + format_interval(time_end - TIME_INVOKED))
