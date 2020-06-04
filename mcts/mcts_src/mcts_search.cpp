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
    float absolute_value =  this->to_play == 1 ? this->value_sum:-this->value_sum;
    return absolute_value / this->visit_count;
}

Node::Node(float prior) {
    this->prior = prior;
    this->value_sum = 0;
    this->visit_count = 0;
    this->reward = 0;
}

PyObject * call_function(PyObject * func,std::string method,bool obj,PyObject * args1,PyObject * args2){
    bool noarg = (args1 == NULL);
    if(noarg){
        args1 = PyDict_New();
    }


    PyObject *keywords = PyDict_New();

    PyObject *myobject_method = PyObject_GetAttrString(func, method.c_str());

    PyObject *result;
    if(obj) {
        if(args2 == NULL) {
            result = PyObject_CallFunctionObjArgs(myobject_method, args1, NULL);
        }else{
            result = PyObject_CallFunctionObjArgs(myobject_method, args1,args2, NULL);
        }
    }else {
        result = PyObject_Call(myobject_method, args1, keywords);
    }

    Py_DECREF(keywords);
    Py_DECREF(myobject_method);
    if(noarg){
        Py_DECREF(args1);
    }
    return result;
}

PyObject * parse_array(PyObject * arr){
    PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(arr);


    int batch_size{ PyArray_SHAPE(np_ret)[0] };
    int height{ PyArray_SHAPE(np_ret)[1] };
    int width{ PyArray_SHAPE(np_ret)[2] };

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

    delete[] c_arr;
    Py_DECREF(np_ret);

    PyObject *pArray = PyArray_SimpleNewFromData(
        3, PyArray_SHAPE(np_ret), NPY_DOUBLE, reinterpret_cast<void*>(ret_np));
    return pArray;
}

void expand_node(shared_ptr<Node> node, int player,PyObject * actions,PyObject * network_output){
    node->to_play = player;
    node->hidden_state = PyObject_GetAttrString(network_output,"hidden_state");

    PyObject * policy_logits = PyObject_GetAttrString(network_output,"policy_logits");
    PyObject * reward_obj = PyObject_GetAttrString(network_output,"reward");
    node->reward = PyFloat_AsDouble(reward_obj);
    Py_DECREF(reward_obj);
    Py_ssize_t action_number = PyList_Size(actions);
    PyObject * one_action;
    PyObject * one_prob;
    for(int i = 0;i < action_number;i ++){
        one_action = PyList_GetItem(actions, i);
        one_prob = PyList_GetItem(policy_logits, i);
        PyObject* one_action_index = PyObject_GetAttrString(one_action,"index");
        int action_int = PyInt_AsLong(one_action_index);
        float action_prob = PyFloat_AsDouble(one_prob);
        node->children[action_int] = make_shared<Node>(action_prob);
        node->children_actions.push_back(action_int);
    }
}

float ucb_score(int pb_c_base,float pb_c_init,shared_ptr<Node> parent,shared_ptr<Node> child){
    float pb_c = log((parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
    pb_c *= (sqrt(parent->visit_count) / (child->visit_count + 1));
    float prior_score = pb_c * child->prior;
    float value_score = child->value();
    if(isnan(value_score)){
        value_score = 0;
    }
    return prior_score + value_score;
}

int select_child(shared_ptr<Node> node,int pb_c_base,float pb_c_init){
    /*
    Select the child with the highest UCB score.
    */
    // TODO 分析程序瓶颈
    // TODO 是否可以用最大堆或其他数据结构优化child的select?
    float current_score,max_score=-INFINITY;
    int max_action = -1;

    for(pair<const int, shared_ptr<Node>> one_child : node->children){
        int one_action = one_child.first;
        shared_ptr<Node> child = one_child.second;
        current_score = ucb_score(pb_c_base,pb_c_init,node,child);
        if(current_score > max_score){
            max_score = current_score;
            max_action = one_action;
        }
    }
    return max_action;
}

void backpropagate(std::vector<shared_ptr<Node>> search_path,float value,int to_play,float discount){
    for(int i = search_path.size() - 1;i >= 0;i --){
        shared_ptr<Node> node = search_path[i];
        node->value_sum += value; //这里的value实际上是区分红黑方的，红方优势趋近于-1，黑方优势趋近于1
        //(node->to_play == to_play? value:-value);
        node->visit_count += 1;
        value = node->reward + discount * value;
    }
}

void add_exploration_noise(float root_dirichlet_alpha,float root_exploration_fraction,shared_ptr<Node> node){
    std::random_device rd;
    std::mt19937 gen(rd());

    vector<double> alphas(node->children.size());
    std::fill(alphas.begin(),alphas.end(),root_dirichlet_alpha);
    dirichlet_distribution<std::mt19937> d(alphas);

    map<int,shared_ptr<Node>>::iterator it;
    it = node->children.begin();
    float child_prior_sum = 0;
    for(auto one_child: node->children){
        child_prior_sum += one_child.second->prior;
    }
    for (double noise : d(gen)) {
        shared_ptr<Node> children = it->second;
        children->prior = children->prior / child_prior_sum * (1 - root_exploration_fraction) + noise * root_exploration_fraction;
        it ++;
    }
}

void run_mcts_cpp(
        PyObject * config,
        PyObject * action_history,
        PyObject * network,
        PyObject * game,
        bool train,
        PyObject * root_py
){
    Py_Initialize();
    _import_array();

    // 创建 root节点
    std::shared_ptr<Node> root = std::make_shared<Node>(0);

    // 利用环境产生当前局面的observation
    PyObject *observation;
    PyObject *args = Py_BuildValue("(i)",-1);
    observation = call_function(game, "make_observation",false, args);
    Py_DECREF(args);

    // 打印observation
    call_function(game,"print_observation_str",NULL);

    // 执行网络前向
    //observation = parse_array(observation);
    //args = Py_BuildValue("(s)",observation);
    PyObject * network_result = call_function(network, "initial_inference", true, observation);
    PyObject * legal_actions = call_function(game, "legal_actions",false, NULL);
    PyObject * action_space = call_function(action_history, "action_space",false, NULL);
    PyObject * to_play = call_function(game, "to_play",false, NULL);

    int root_player;
    PyObject* player_obj = PyObject_GetAttrString(to_play,"player");
    root_player = PyLong_AsLong(player_obj);
    Py_DECREF(player_obj);

    // 执行蒙特卡洛树根节点展开
    expand_node(root, root_player,legal_actions,network_result);
    // TODO 查看mcts树是否符合预期

    Py_DECREF(observation);
    Py_DECREF(network_result);
    Py_DECREF(legal_actions);
    Py_DECREF(to_play);

    PyObject * sim_time_obj = PyObject_GetAttrString(config,"num_simulations");
    int sim_times = PyInt_AsLong(sim_time_obj);
    Py_DECREF(sim_time_obj);

    PyObject * discount_obj = PyObject_GetAttrString(config,"discount");
    float discount = PyFloat_AsDouble(discount_obj);
    Py_DECREF(discount_obj);

    PyObject * root_exploration_fraction_obj = PyObject_GetAttrString(config,"root_exploration_fraction");
    float root_exploration_fraction = PyFloat_AsDouble(root_exploration_fraction_obj);
    Py_DECREF(root_exploration_fraction_obj);

    PyObject * root_dirichlet_alpha_obj = PyObject_GetAttrString(config,"root_dirichlet_alpha");
    float root_dirichlet_alpha = PyFloat_AsDouble(root_dirichlet_alpha_obj);
    Py_DECREF(root_dirichlet_alpha_obj);

    add_exploration_noise(root_exploration_fraction,root_dirichlet_alpha,root);

    //初始化search path等等

    for(int simulate_id = 0;simulate_id < sim_times;simulate_id ++){
        std::vector<shared_ptr<Node>> search_path;
        shared_ptr<Node> node = root;
        search_path.push_back(root);
        int current_player = root_player;

        int action_selected;
        while(node->expanded()){
            action_selected = select_child(node);
            node = node->children[action_selected];
            current_player = 1 - current_player;
            search_path.push_back(node);
        }
        shared_ptr<Node> parent = search_path[search_path.size() - 2];

        PyObject* action =  PyInt_FromLong(action_selected);
        network_result = call_function(network, "recurrent_inference",true,parent->hidden_state,action);
        expand_node(node, current_player,action_space,network_result);

        PyObject * value_obj = PyObject_GetAttrString(network_result,"value");
        float value = PyFloat_AsDouble(value_obj);
        backpropagate(search_path,value,current_player,discount);
        Py_DECREF(network_result);
        Py_DECREF(action);
        Py_DECREF(value_obj);
    }
    Py_DECREF(action_space);

    set_node(root_py,root);
}

void set_node(PyObject * root,shared_ptr<Node> node_cpp,int depth){
    PyObject_SetAttrString(root,"visit_count",PyInt_FromLong(node_cpp->visit_count));
    PyObject_SetAttrString(root,"to_play",PyInt_FromLong(node_cpp->to_play));
    PyObject_SetAttrString(root,"prior",PyFloat_FromDouble(node_cpp->prior));
    PyObject_SetAttrString(root,"value_sum",PyFloat_FromDouble(node_cpp->value_sum));
    PyObject_SetAttrString(root,"reward",PyFloat_FromDouble(node_cpp->reward));
    if(depth <= 0){
        return;
    }
    PyObject * children_dict = PyObject_GetAttrString(root,"children");
    for(auto entry:node_cpp->children){
        int action_id = entry.first;
        shared_ptr<Node> node = entry.second;

        PyObject * one_node = call_function(root,"clone",NULL);
        PyDict_SetItemString(children_dict,std::to_string(action_id).c_str(),one_node);
        set_node(one_node,node,depth - 1);
    }
}
