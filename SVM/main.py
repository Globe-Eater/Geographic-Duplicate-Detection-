#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:26:51 2019

@author: kellenbullock

This programming will call all the functions and proccess.
"""

import preprocessing
import mapping

class Proccess(object):
    
    def go(self):
        # This is just to help control the order of operations
        exit(1)

class Engine(object):
    
    def __init__(self, proccess_map):
        '''This class is what drives the program forward.
        Please reference the Order class below '''
        self.proccess_map = proccess_map
    
    def start(self):
        current_proccess = self.Order.preproccessing()
        
        while True:
            print("/n")
            next_proccess_name = current_proccess.go()
            current_proccess = self.proccess_map.next_proccess(next_proccess_name)

# going to come back to this need to change all the scenes to proccess names
class Order(object):

    steps = {
    'preproccessing': preproccessing(),
    'map': mapping(),
    'score': metric(),
    'SVM': SVM(),
    }

    def __init__(self, start_proccess):
        self.start_proccess = start_proccess

    def next_proccess(self, proccess_name):
        return Order.proccess.get(proccess_name)

    def preproccessing(self):
        return self.next_proccess(self.start_proccess)

# This operation uses the object Proccess and the method called start to commence the program.
Proccess.start()