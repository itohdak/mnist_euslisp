#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    log_0_001 = np.loadtxt('./log_0_001.txt')
    log_0_002 = np.loadtxt('./log_0_002.txt')
    log_0_003 = np.loadtxt('./log_0_003.txt')
    x1 = np.arange(0, len(log_0_001), 1.0)
    x2 = np.arange(0, len(log_0_002), 1.0)
    x3 = np.arange(0, len(log_0_003), 1.0)
    plt.plot(x1, log_0_001, "r")
    # plt.plot(x2, log_0_002, "g")
    # plt.plot(x3, log_0_003, "b")
    plt.show()
