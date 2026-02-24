#include "Rendering/animation/AnimationPlayer.h"
#include "Resource/model/Model.h"

bool AnimationPlayer::update(float deltaTime)
{
    if (!m_model) {
        return false;
    }
    const auto& animations = m_model->getAnimations();
    if (animations.empty() || m_activeIndex >= animations.size()) {
        return false;
    }
    return m_model->updateAnimation(m_activeIndex, deltaTime);
}
