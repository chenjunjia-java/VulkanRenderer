#pragma once

// System
#include <cstdint>

class Model;

class AnimationPlayer {
public:
    AnimationPlayer() = default;
    explicit AnimationPlayer(Model* model) : m_model(model) {}

    void setModel(Model* model) { m_model = model; }
    Model* getModel() const { return m_model; }

    void setActiveAnimation(uint32_t index) { m_activeIndex = index; }
    uint32_t getActiveAnimation() const { return m_activeIndex; }

    /// Returns true if any node transform was modified (for TLAS invalidation).
    bool update(float deltaTime);

private:
    Model* m_model = nullptr;
    uint32_t m_activeIndex = 0;
};
