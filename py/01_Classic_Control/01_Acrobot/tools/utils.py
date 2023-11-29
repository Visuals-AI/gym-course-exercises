#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : EXP
# @Time   : 2023/11/25 23:56
# -----------------------------------------------

import os
import time


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



def current_millis() :
    return int(time.time() * 1000)

