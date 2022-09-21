# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 9/19/2022

import random
import numpy as np

_delta = 15
alpha = lambda alphal: alphal + random.randint(0, _delta) if (alphal <= 255-_delta) \
    else alphal + random.randint(-_delta, 0)
brighness = lambda x: np.array(x)/(random.random()*3)

yellow = brighness([alpha(0), alpha(255), alpha(255)]).astype('uint8')
red = brighness([alpha(0), alpha(0), alpha(255)]).astype('uint8')
blue = brighness([alpha(255), alpha(0), alpha(0)]).astype('uint8')
green = brighness([alpha(0), alpha(255), alpha(0)]).astype('uint8')
gray = brighness([alpha(255), alpha(255), alpha(255)]).astype('uint8')
p = brighness([alpha(255), alpha(255), alpha(0)]).astype('uint8')

color = [red, [alpha(255), alpha(255), alpha(255)]]
