#include "ECS/component/Component.h"

Component::~Component()
{
    if (state != State::Destroyed) {
        OnDestroy();
        state = State::Destroyed;
    }
}

void Component::Initialize()
{
    if (state == State::Uninitialized) {
        state = State::Initializing;
        OnInitialize();
        state = State::Active;
    }
}

void Component::Destroy()
{
    if (state == State::Active) {
        state = State::Destroying;
        OnDestroy();
        state = State::Destroyed;
    }
}
