import numpy as np
cimport numpy as np
cimport cython
import time
import random
import math
from libcpp cimport bool


cdef extern from "mcts_search.cpp":
    float plus(float a,float b);
    void say_hello_cpp(object obj);
    void run_mcts_cpp(
        object config,
        object action_history,
        object network,
        object game,
        bool train
    );


def plus_test(a,b):
    return plus(a,b)

def run_mcts(config,action_history,network,game,train):
    return run_mcts_cpp(config,action_history,network,game,train)


