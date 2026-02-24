#pragma once

// System
#include <string>
#include <vector>

enum class AnimationInterpolation {
    Linear,
    Step,
    CubicSpline,
};

enum class AnimationPath {
    Translation,
    Rotation,
    Scale,
    Weights,
};

struct AnimationSampler {
    AnimationInterpolation interpolation = AnimationInterpolation::Linear;

    // Keyframe time values (seconds).
    std::vector<float> inputs;

    // Packed output values; `outputComponents` indicates vec3/vec4 layout.
    std::vector<float> outputs;
    int outputComponents = 0;
};

struct AnimationChannel {
    int samplerIndex = -1;
    int targetNode = -1;
    AnimationPath path = AnimationPath::Translation;
};

struct Animation {
    std::string name;
    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;

    float start = 0.0f;
    float end = 0.0f;
    float currentTime = 0.0f;
};

