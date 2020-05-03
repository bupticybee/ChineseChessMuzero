#include "mcts_search.hpp"

float plus(float a, float b){
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

PyObject * call_function(PyObject * game,std::string method,PyObject * args){


    //PyObject_CallMethod(obj,"run","call_param");
    //PyObject *args = Py_BuildValue("(i)",position);

    PyObject *keywords = PyDict_New();

    //PyDict_SetItemString(keywords, "somearg", Py_True);

    PyObject *myobject_method = PyObject_GetAttrString(game, method.c_str());


    PyObject *result = PyObject_Call(myobject_method, args, keywords);

    //PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(result);


    /*
    int batch_size{ PyArray_SHAPE(np_ret)[0] };
    int height{ PyArray_SHAPE(np_ret)[1] };
    int width{ PyArray_SHAPE(np_ret)[2] };

    long * ret_np;

    ret_np = reinterpret_cast<long *>(PyArray_DATA(np_ret));

    std::cout << "shape like:  " << batch_size << " " << height << " " << width << std::endl;
    for (int i{}; i < batch_size; i++)
        std::cout << ret_np[i] << ' ';
    */

    Py_DECREF(keywords);
    Py_DECREF(myobject_method);
    //Py_DECREF(result);
    //Py_DECREF(np_ret);
    return result;
}

static void reprint(PyObject *obj) {
    PyObject* repr = PyObject_Repr(obj);
    PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    const char *bytes = PyBytes_AS_STRING(str);

    printf("%s\n", bytes);

    Py_XDECREF(repr);
    Py_XDECREF(str);
}

void run_mcts_cpp(
        PyObject * config,
        PyObject * action_history,
        PyObject * network,
        PyObject * game,
        bool train
){
    std::shared_ptr<Node> root = std::make_shared<Node>(0);
    PyObject *result;

    PyObject *args = Py_BuildValue("(i)",-1);
    result = call_function(game,"make_observation",args);
    Py_DECREF(args);
    Py_DECREF(result);

    args = PyDict_New();
    result = call_function(game,"print_observation_str",args);
    Py_DECREF(args);

}

