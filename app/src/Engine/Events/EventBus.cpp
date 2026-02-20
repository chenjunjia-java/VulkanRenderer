#include "Engine/Events/EventBus.h"

#include <algorithm>

uint64_t EventBus::subscribeImpl(std::type_index type, std::function<void(const void*)>&& fn)
{
    uint64_t id = nextListenerId++;
    listenersByType[type].push_back(Listener{ id, true, std::move(fn) });
    return id;
}

void EventBus::unsubscribe(std::type_index type, uint64_t id)
{
    auto it = listenersByType.find(type);
    if (it == listenersByType.end()) return;

    ListenerList& list = it->second;

    // If we're currently publishing (possibly nested), mark as dead and compact later.
    if (publishDepth > 0) {
        for (auto& l : list) {
            if (l.id == id && l.alive) {
                l.alive = false;
                needsCompaction = true;
                break;
            }
        }
        return;
    }

    list.erase(std::remove_if(list.begin(), list.end(),
                              [id](const Listener& l) { return l.id == id; }),
               list.end());
}

void EventBus::compactDeadListeners(std::type_index type)
{
    auto it = listenersByType.find(type);
    if (it == listenersByType.end()) return;

    ListenerList& list = it->second;
    list.erase(std::remove_if(list.begin(), list.end(),
                              [](const Listener& l) { return !l.alive; }),
               list.end());
}

void EventBus::publishImpl(std::type_index type, const void* payload)
{
    auto it = listenersByType.find(type);
    if (it == listenersByType.end()) return;

    ListenerList& list = it->second;

    publishDepth++;
    for (auto& l : list) {
        if (!l.alive) continue;
        if (l.fn) {
            l.fn(payload);
        }
    }
    publishDepth--;

    if (publishDepth == 0 && needsCompaction) {
        // Compact all types to keep things simple for the minimal version.
        for (auto& kv : listenersByType) {
            ListenerList& lst = kv.second;
            lst.erase(std::remove_if(lst.begin(), lst.end(),
                                     [](const Listener& l) { return !l.alive; }),
                      lst.end());
        }
        needsCompaction = false;
    }
}

void EventBus::process()
{
    if (queued.empty()) return;

    // Allow enqueue() during processing without re-entrancy issues.
    std::vector<std::function<void()>> current;
    current.swap(queued);

    for (auto& task : current) {
        if (task) {
            task();
        }
    }
}

