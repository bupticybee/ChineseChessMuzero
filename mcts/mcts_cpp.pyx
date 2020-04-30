import numpy as np
cimport numpy as np
cimport cython
import time
import random
import math

cdef extern from "mcts_search.cpp":
    float plus(float a,float b);
    void say_hello_cpp(object obj);

def plus_test(a,b):
    return plus(a,b)

def run_mcts(config, root, action_history, network):
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    # TODO 这里乱clone还得了，太慢了,这个必须优化
    for _ in range(config.num_simulations):
        t0 = time.time()
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            #action, node = select_child(config, node)
            #history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        t1 = time.time()
        network_output = network.recurrent_inference(parent.hidden_state, history.last_action())
        t2 = time.time()
        #expand_node(node, history.to_play(), history.action_space(), network_output)

        #backpropagate(search_path, network_output.value, history.to_play(), config.discount, min_max_stats)
        t3 = time.time()
        print("cpu time",t1 - t0 + t3 - t2)
        print("gpu time",t2 - t1)


def say_hello(input_class):
    print("result is :",say_hello_cpp(input_class))

def select_child(config, node):
    """
    Select the child with the highest UCB score.
    """
    # When the parent visit count is zero, all ucb scores are zeros, therefore we return a random child
    if node.visit_count == 0:
        return random.sample(node.children.items(), 1)[0]

    _, action, child = max(
        (ucb_score(config, node, child), action,
         child) for action, child in node.children.items())
    return action, child


def ucb_score(config, parent, child,
              ) -> float:
    """
    The score for a node is based on its value, plus an exploration bonus based on
    the prior.
    """
    #pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    #pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    #prior_score = pb_c * child.prior
    value_score = child.value()
    #return prior_score + value_score
    return value_score
