#pragma once

#include <cstdint>
#include <functional>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

// Minimal, type-safe EventBus:
// - subscribe<T>(handler) returns a move-only Subscription (RAII auto-unsubscribe)
// - publish<T>(event) dispatches immediately
// - enqueue<T>(event) queues a copy/move; process() dispatches queued events later (e.g. end-of-frame)
//
// Notes:
// - First version is single-thread oriented.
// - Unsubscribe during dispatch is safe (listener is marked dead and compacted after dispatch).
class EventBus {
public:
    class Subscription {
    public:
        Subscription() = default;
        Subscription(EventBus* inBus, std::type_index inType, uint64_t inId)
            : bus(inBus)
            , type(inType)
            , id(inId)
        {
        }

        Subscription(const Subscription&) = delete;
        Subscription& operator=(const Subscription&) = delete;

        Subscription(Subscription&& other) noexcept
        {
            *this = std::move(other);
        }
        Subscription& operator=(Subscription&& other) noexcept
        {
            if (this == &other) return *this;
            reset();
            bus = other.bus;
            type = other.type;
            id = other.id;
            other.bus = nullptr;
            other.type = std::type_index(typeid(void));
            other.id = 0;
            return *this;
        }

        ~Subscription() { reset(); }

        void reset()
        {
            if (bus) {
                bus->unsubscribe(type, id);
            }
            bus = nullptr;
            type = std::type_index(typeid(void));
            id = 0;
        }

        explicit operator bool() const { return bus != nullptr; }

    private:
        EventBus* bus = nullptr;
        std::type_index type = std::type_index(typeid(void));
        uint64_t id = 0;
    };

    EventBus() = default;
    ~EventBus() = default;

    EventBus(const EventBus&) = delete;
    EventBus& operator=(const EventBus&) = delete;
    EventBus(EventBus&&) = delete;
    EventBus& operator=(EventBus&&) = delete;

    template <typename E, typename F>
    Subscription subscribe(F&& handler)
    {
        std::type_index type = std::type_index(typeid(E));
        auto erased = [fn = std::forward<F>(handler)](const void* payload) mutable {
            fn(*static_cast<const E*>(payload));
        };
        uint64_t id = subscribeImpl(type, std::move(erased));
        return Subscription(this, type, id);
    }

    template <typename E>
    void publish(const E& event)
    {
        publishImpl(std::type_index(typeid(E)), &event);
    }

    template <typename E>
    void enqueue(E event)
    {
        // Store a callable that owns the event by value and publishes later.
        queued.emplace_back([this, ev = std::move(event)]() mutable { this->publish(ev); });
    }

    // Dispatch all queued events (FIFO). Safe to call once per frame.
    void process();

    void clearQueue() { queued.clear(); }
    size_t queuedCount() const { return queued.size(); }

private:
    struct Listener {
        uint64_t id = 0;
        bool alive = true;
        std::function<void(const void*)> fn;
    };

    using ListenerList = std::vector<Listener>;

    uint64_t subscribeImpl(std::type_index type, std::function<void(const void*)>&& fn);
    void unsubscribe(std::type_index type, uint64_t id);
    void publishImpl(std::type_index type, const void* payload);
    void compactDeadListeners(std::type_index type);

    std::unordered_map<std::type_index, ListenerList> listenersByType;
    std::vector<std::function<void()>> queued;

    uint64_t nextListenerId = 1;
    int publishDepth = 0;
    bool needsCompaction = false;
};

