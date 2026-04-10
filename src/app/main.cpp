#include <vesta/render/vulkan/vk_engine.h>

int main(int argc, char* argv[])
{
    // Keep main intentionally small: all interesting lifetime management happens
    // inside VestaEngine so startup and shutdown order stays explicit.
    VestaEngine engine;

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}
