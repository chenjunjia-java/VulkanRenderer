#pragma once

#include <cstddef>
#include <type_traits>

// Forward declaration for owner reference
class Entity;

// Component type ID system for O(1) lookup
class ComponentTypeIDSystem {
private:
    static inline size_t nextTypeID = 0;

public:
    template <typename T>
    static size_t GetTypeID() {
        static size_t typeID = nextTypeID++;
        return typeID;
    }
};

// Base component class with lifecycle management
class Component {
public:
    enum class State {
        Uninitialized,
        Initializing,
        Active,
        Destroying,
        Destroyed
    };

    virtual ~Component();

    void Initialize();
    void Destroy();

    bool IsActive() const { return state == State::Active; }

    void SetOwner(Entity* entity) { owner = entity; }
    Entity* GetOwner() const { return owner; }

    template <typename T>
    static size_t GetTypeID() {
        return ComponentTypeIDSystem::GetTypeID<T>();
    }

protected:
    virtual void OnInitialize() {}
    virtual void OnDestroy() {}
    virtual void Update(float deltaTime) { (void)deltaTime; }
    virtual void Render() {}

    State state = State::Uninitialized;
    Entity* owner = nullptr;

    friend class Entity;
};
