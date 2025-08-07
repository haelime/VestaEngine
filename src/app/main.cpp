#include <vesta/render/vulkan/vk_engine.h>

int main(int argc, char* argv[])
{
    VestaEngine engine;

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}
