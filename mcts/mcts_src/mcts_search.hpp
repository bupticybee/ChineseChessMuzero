#ifndef MCTS_H
#define MCTS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <iostream>
#include <vector>
#include <Python.h>
#include <memory>
#include <numpy/ndarraytypes.h>
using namespace std;

float pluscpp(float a, float b);

void say_hello_cpp(PyObject * obj);

class Node{
    public:
        Node(float prior);
        int visit_count;
        int to_play;
        float prior;
        float value_sum;
        std::vector<Node*> children;
        PyObject * hidden_state;
        double reward;
        bool expanded();
        float value();
};

void expand_node(shared_ptr<Node> node, PyObject * to_play,PyObject * actions,PyObject * network_output);

PyObject * call_function(PyObject * func,std::string method,PyObject * args= NULL,bool obj=false);
static void reprint(PyObject *obj) ;

void run_mcts_cpp(
        PyObject * config,
        PyObject * action_history,
        PyObject * network,
        PyObject * game,
        bool train
        );

PyObject* parse_array(PyObject* arr);

#endif