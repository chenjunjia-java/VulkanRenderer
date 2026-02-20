#include <iostream>

#include "Runtime/VulkanApplication.h"
#include "ECS/ECS.h"

// Minimal verification of component system (can be removed after validation)
namespace {
struct TestComponent : public Component {
    int value = 0;
    void OnInitialize() override { value = 42; }
    void Update(float dt) override { (void)dt; value += 1; }
};
}  // namespace

int main() {
    // Verify ECS component system
    {
        Scene scene;
        Entity* entity = scene.AddEntity("TestEntity");
        TestComponent* comp = entity->AddComponent<TestComponent>();
        if (!comp || entity->GetComponent<TestComponent>() != comp) {
            std::cerr << "ECS verification failed: GetComponent mismatch\n";
            return EXIT_FAILURE;
        }
        scene.Initialize();
        if (comp->value != 42) {
            std::cerr << "ECS verification failed: OnInitialize not called\n";
            return EXIT_FAILURE;
        }
        scene.Update(0.0f);
        if (comp->value != 43) {
            std::cerr << "ECS verification failed: Update not called\n";
            return EXIT_FAILURE;
        }
    }
    VulkanApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}