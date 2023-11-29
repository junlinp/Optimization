#include "src/CGraph.h"
using namespace CGraph;
class MyNode1 : public CGraph::GNode {
public:
    CStatus run() override {
        printf("[%s], Sleep for 1 second ...\n", this->getName().c_str());
        CGRAPH_SLEEP_SECOND(1)
        return CStatus();
    }
};

class MyNode2 : public CGraph::GNode {
public:
    CStatus run() override {
        printf("[%s], Sleep for 2 second ...\n", this->getName().c_str());
        CGRAPH_SLEEP_SECOND(2)
        return CStatus();
    }
};

int main(int argc, char** argv) {
    GPipelinePtr pipeline = GPipelineFactory::create();
    GElementPtr a, b, c, d = nullptr;

    pipeline->registerGElement<MyNode1>(&a, {}, "nodeA");
    pipeline->registerGElement<MyNode2>(&b, {a}, "nodeB");
    pipeline->registerGElement<MyNode1>(&c, {a}, "nodeC");
    pipeline->registerGElement<MyNode2>(&d, {b, c}, "nodeD");

    GFunctionPtr function_node = nullptr;
    pipeline->registerGElement<GFunction>(&function_node, {d});
    function_node->setFunction(CFunctionType::RUN, []() {
        printf("Do nothin");
        return CStatus();
    });

    pipeline->process();
    GPipelineFactory::remove(pipeline);
}