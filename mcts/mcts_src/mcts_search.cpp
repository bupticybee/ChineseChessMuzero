#include "mcts_search.hpp"

float pluscpp(float a, float b){
    return a + b;
}

void say_hello_cpp(PyObject * obj){
    std::cout << "saying hello" << std::endl;

    //PyObject_CallMethod(obj,"run","call_param");
    PyObject *args = Py_BuildValue("(s)","blahdy blah haha");
    PyObject *keywords = PyDict_New();
    //PyDict_SetItemString(keywords, "somearg", Py_True);

    PyObject *myobject_method = PyObject_GetAttrString(obj, "run");

    PyObject *result = PyObject_Call(myobject_method, args, keywords);
    Py_DECREF(args);
    Py_DECREF(keywords);
    Py_DECREF(myobject_method);
    Py_DECREF(result);

}

bool Node::expanded(){
    return this->children.size() > 0;
}

float Node::value(){
    //PyObject * obj;
    if(this->visit_count == 0){
        return NAN;
    }
    return this->value_sum / this->visit_count;
}

Node::Node(float prior) {
    this->prior = prior;
}

PyObject * call_function(PyObject * func,std::string method,PyObject * args,bool obj){
    bool noarg = (args == NULL);
    if(noarg){
        args = PyDict_New();
    }


    PyObject *keywords = PyDict_New();

    PyObject *myobject_method = PyObject_GetAttrString(func, method.c_str());

    PyObject *result;
    if(obj) {
        result = PyObject_CallFunctionObjArgs(myobject_method, args,NULL);
    }else {
        result = PyObject_Call(myobject_method, args, keywords);
    }

    Py_DECREF(keywords);
    Py_DECREF(myobject_method);
    if(noarg){
        Py_DECREF(args);
    }
    return result;
}

PyObject * parse_array(PyObject * arr){
    PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(arr);


    int batch_size{ PyArray_SHAPE(np_ret)[0] };
    int height{ PyArray_SHAPE(np_ret)[1] };
    int width{ PyArray_SHAPE(np_ret)[2] };

    // TODO free memory space
    double* c_arr = new double[batch_size * height * width];

    double * ret_np;

    ret_np = reinterpret_cast<double *>(PyArray_DATA(np_ret));

    std::cout << "shape like:  " << batch_size << " " << height << " " << width << std::endl;
    for (int i=0; i < batch_size; i++){
        for (int j=0; j < height; j++){
            for (int k=0; k < width; k++){
                std::cout << ret_np[i * (height * width) + j * width + k] << "\t";
                c_arr[i * (height * width) + j * width + k] = ret_np[i * (height * width) + j * width + k];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

    std::cout << __LINE__ << std::endl;
    Py_DECREF(np_ret);

    std::cout << __LINE__ << std::endl;
    PyObject *pArray = PyArray_SimpleNewFromData(
        3, PyArray_SHAPE(np_ret), NPY_DOUBLE, reinterpret_cast<void*>(ret_np));
    return pArray;
}

void expand_node(shared_ptr<Node> node, PyObject * to_play,PyObject * actions,PyObject * network_output){
    //TODO finish here
    long player;
    PyObject* player_obj = PyObject_GetAttrString(to_play,"player");
    player = PyLong_AsLong(player_obj);
    Py_DECREF(player_obj);
    node->to_play = player;
    node->hidden_state = PyObject_GetAttrString(network_output,"hidden_state");
    PyObject * reward_obj = PyObject_GetAttrString(network_output,"reward");
    node->reward = PyFloat_AsDouble(reward_obj);
    Py_DECREF(reward_obj);
    // TODO finish here
}

void run_mcts_cpp(
        PyObject * config,
        PyObject * action_history,
        PyObject * network,
        PyObject * game,
        bool train
){
    Py_Initialize();
    _import_array();

    // 创建 root节点
    std::shared_ptr<Node> root = std::make_shared<Node>(0);

    // 利用环境产生当前局面的observation
    PyObject *observation;
    PyObject *args = Py_BuildValue("(i)",-1);
    observation = call_function(game, "make_observation", args);
    Py_DECREF(args);

    // 打印observation
    call_function(game,"print_observation_str",NULL);

    // 执行网络前向
    //observation = parse_array(observation);
    //args = Py_BuildValue("(s)",observation);
    PyObject * network_result = call_function(network, "initial_inference", observation, true);
    PyObject * legal_actions = call_function(game, "legal_actions", NULL);
    PyObject * to_play = call_function(game, "to_play", NULL);

    expand_node(root, to_play,legal_actions,network_result);

    Py_DECREF(observation);
    Py_DECREF(network_result);
    Py_DECREF(legal_actions);
    Py_DECREF(to_play);

}

