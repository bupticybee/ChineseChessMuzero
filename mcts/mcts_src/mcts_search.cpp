#include "mcts_search.hpp"

float plus(float a, float b){
    return a + b;
}

void say_hello_cpp(PyObject * obj){
    std::cout << "saying hello" << std::endl;

    //PyObject_CallMethod(obj,"run","call_param");
    PyObject *args = Py_BuildValue("(s)","blahdy blah");
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

