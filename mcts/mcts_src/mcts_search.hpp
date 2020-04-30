#ifndef MCTS_H
#define MCTS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <iostream>
#include <vector>
#include <Python.h>

float plus(float a, float b);

void say_hello_cpp(PyObject * obj);

class Node{
    public:
        int visit_count;
        int to_play;
        float prior;
        float value_sum;
        std::vector<Node*> children;
        void* hidden_state;
        float reward;
        Node(float prior);
        bool expanded();
        float value();
};

#endif
