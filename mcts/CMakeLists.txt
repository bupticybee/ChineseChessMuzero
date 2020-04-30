# 指定cmake最低编译版本
CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
# 指定工程的名称
PROJECT(MCTSCPP)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
#指定头文件目录位置
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/mcts_src)
#添加共享库搜索路径
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)
#生成可执行文件
ADD_EXECUTABLE(mcts mcts_src/mcts_search.cpp)
target_link_libraries(mcts ${PYTHON_LIBRARIES})
