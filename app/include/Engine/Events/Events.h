#pragma once

// Keep event payloads small and data-only.

struct FramebufferResizeEvent {
    int width = 0;
    int height = 0;
};

