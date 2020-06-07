#ifndef MCTS_H
#define MCTS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <iostream>
#include <vector>
#include <map>
#include <Python.h>
#include <memory>
#include "dirichlet.hpp"
#include <random>
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
        std::map<int,shared_ptr<Node>> children;
        PyObject * hidden_state;
        double reward;
        bool expanded();
        float value();
};

void expand_node(shared_ptr<Node> node, int player,PyObject * actions,PyObject * network_output);
float ucb_score(int pb_c_base,float pb_c_init,shared_ptr<Node> parent,shared_ptr<Node> child);
int select_child(shared_ptr<Node> node,int pb_c_base=19652,float pb_c_init=1.25);
void backpropagate(std::vector<shared_ptr<Node>> search_path,float value,int to_play,float discount);

PyObject * call_function(PyObject * func,std::string method,bool obj,PyObject * args1= NULL,PyObject * args2=NULL);
static void reprint(PyObject *obj) ;

void run_mcts_cpp(
        PyObject * config,
        PyObject * action_history,
        PyObject * network,
        PyObject * game,
        bool train,
        PyObject * root_py
        );

PyObject* parse_array(PyObject* arr);

void set_node(PyObject * root,shared_ptr<Node> node_cpp,int depth=1);

void add_exploration_noise(float root_dirichlet_alpha,float root_exploration_fraction,shared_ptr<Node> node);

#endif