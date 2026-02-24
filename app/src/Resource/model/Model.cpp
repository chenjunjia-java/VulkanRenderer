#include "Resource/model/Model.h"

// System
#include <algorithm>
#include <filesystem>
#include <functional>
#include <limits>

// Third-party
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

// Project
#include "Configs/AppConfig.h"
#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Resource/model/loaders/GltfModelLoader.h"
#include "Resource/model/loaders/ObjModelLoader.h"

glm::mat4 Node::getLocalMatrix() const
{
    // glTF semantics: use either matrix OR TRS (matrix overrides TRS).
    if (hasMatrix) {
        return matrix;
    }
    // Match tutorial order: T * R * S
    return glm::translate(glm::mat4(1.0f), translation)
        * glm::mat4_cast(rotation)
        * glm::scale(glm::mat4(1.0f), scale);
}

glm::mat4 Node::getGlobalMatrix() const
{
    glm::mat4 m = getLocalMatrix();
    const Node* p = parent;
    while (p) {
        m = p->getLocalMatrix() * m;
        p = p->parent;
    }
    return m;
}

Node* Model::findNode(const std::string& name) const
{
    for (Node* n : linearNodes) {
        if (n && n->name == name) {
            return n;
        }
    }
    return nullptr;
}

Node* Model::getNodeByGltfIndex(size_t index) const
{
    if (index >= ownedNodes.size()) {
        return nullptr;
    }
    return ownedNodes[index].get();
}

bool Model::updateAnimation(uint32_t index, float deltaTime)
{
    if (animations.empty() || index >= animations.size()) {
        return false;
    }

    Animation& anim = animations[index];
    anim.currentTime += deltaTime;
    while (anim.currentTime >= anim.end && anim.end > anim.start) {
        anim.currentTime -= (anim.end - anim.start);
    }

    const int comps = 3; // vec3 stride for translation/scale
    const int quatComps = 4;
    bool modified = false;

    for (const AnimationChannel& channel : anim.channels) {
        if (channel.samplerIndex < 0
            || static_cast<size_t>(channel.samplerIndex) >= anim.samplers.size()
            || channel.targetNode < 0) {
            continue;
        }

        Node* node = getNodeByGltfIndex(static_cast<size_t>(channel.targetNode));
        if (!node) {
            continue;
        }

        const AnimationSampler& sampler = anim.samplers[static_cast<size_t>(channel.samplerIndex)];
        if (sampler.inputs.size() < 2 || sampler.outputs.empty()) {
            continue;
        }

        auto nextIt = std::lower_bound(sampler.inputs.begin(), sampler.inputs.end(), anim.currentTime);
        if (nextIt == sampler.inputs.end() || nextIt == sampler.inputs.begin()) {
            continue;
        }

        const size_t i = static_cast<size_t>(std::distance(sampler.inputs.begin(), nextIt) - 1);
        const float t = (anim.currentTime - sampler.inputs[i])
            / (sampler.inputs[i + 1] - sampler.inputs[i]);

        switch (channel.path) {
        case AnimationPath::Translation: {
            if (sampler.outputComponents >= comps && (i + 1) * static_cast<size_t>(comps) <= sampler.outputs.size()) {
                glm::vec3 start(sampler.outputs[i * comps + 0],
                               sampler.outputs[i * comps + 1],
                               sampler.outputs[i * comps + 2]);
                glm::vec3 end(sampler.outputs[(i + 1) * comps + 0],
                             sampler.outputs[(i + 1) * comps + 1],
                             sampler.outputs[(i + 1) * comps + 2]);
                node->translation = glm::mix(start, end, t);
                node->hasMatrix = false;
                modified = true;
            }
            break;
        }
        case AnimationPath::Rotation: {
            if (sampler.outputComponents >= quatComps
                && (i + 1) * static_cast<size_t>(quatComps) <= sampler.outputs.size()) {
                // glTF: xyzw -> glm::quat(w, x, y, z)
                glm::quat start(sampler.outputs[i * quatComps + 3],
                                sampler.outputs[i * quatComps + 0],
                                sampler.outputs[i * quatComps + 1],
                                sampler.outputs[i * quatComps + 2]);
                glm::quat end(sampler.outputs[(i + 1) * quatComps + 3],
                              sampler.outputs[(i + 1) * quatComps + 0],
                              sampler.outputs[(i + 1) * quatComps + 1],
                              sampler.outputs[(i + 1) * quatComps + 2]);
                node->rotation = glm::normalize(glm::slerp(start, end, t));
                node->hasMatrix = false;
                modified = true;
            }
            break;
        }
        case AnimationPath::Scale: {
            if (sampler.outputComponents >= comps && (i + 1) * static_cast<size_t>(comps) <= sampler.outputs.size()) {
                glm::vec3 start(sampler.outputs[i * comps + 0],
                               sampler.outputs[i * comps + 1],
                               sampler.outputs[i * comps + 2]);
                glm::vec3 end(sampler.outputs[(i + 1) * comps + 0],
                             sampler.outputs[(i + 1) * comps + 1],
                             sampler.outputs[(i + 1) * comps + 2]);
                node->scale = glm::mix(start, end, t);
                node->hasMatrix = false;
                modified = true;
            }
            break;
        }
        default:
            break;
        }
    }
    return modified;
}

void Model::clear()
{
    ownedNodes.clear();
    nodes.clear();
    linearNodes.clear();
    textures.clear();
    materials.clear();
    meshes.clear();
    skins.clear();
    animations.clear();
}

void Model::rebuildLinearNodes()
{
    linearNodes.clear();
    linearNodes.reserve(ownedNodes.size());

    std::function<void(Node*)> visit = [&](Node* n) {
        if (!n) return;
        linearNodes.push_back(n);
        n->linearIndex = static_cast<uint32_t>(linearNodes.size() - 1);
        for (Node* c : n->children) {
            visit(c);
        }
    };

    for (Node* root : nodes) {
        visit(root);
    }
}

namespace {
BoundingBox makeEmptyBounds()
{
    const float inf = std::numeric_limits<float>::infinity();
    return BoundingBox(glm::vec3(inf), glm::vec3(-inf));
}

void unionInto(BoundingBox& dst, bool& dstValid, const BoundingBox& src, bool srcValid)
{
    if (!srcValid) return;
    if (!dstValid) {
        dst = src;
        dstValid = true;
        return;
    }
    dst.min = glm::min(dst.min, src.min);
    dst.max = glm::max(dst.max, src.max);
}
} // namespace

void Model::rebuildBounds()
{
    // Clear previous bounds flags.
    for (Node* n : linearNodes) {
        if (!n) continue;
        n->hasSubtreeBounds = false;
        n->subtreeBounds = BoundingBox{};
    }

    std::function<void(Node*)> compute = [&](Node* n) {
        if (!n) return;

        BoundingBox acc = makeEmptyBounds();
        bool valid = false;

        // Local meshes (node space).
        for (uint32_t meshIndex : n->meshIndices) {
            if (meshIndex >= meshes.size()) continue;
            const Mesh& m = meshes[meshIndex];
            unionInto(acc, valid, m.bounds, m.hasBounds);
        }

        // Children (transform their subtree bounds into this node space).
        for (Node* c : n->children) {
            compute(c);
            if (!c || !c->hasSubtreeBounds) continue;

            BoundingBox childBox = c->subtreeBounds;
            childBox.Transform(c->getLocalMatrix());
            unionInto(acc, valid, childBox, true);
        }

        n->hasSubtreeBounds = valid;
        if (valid) {
            n->subtreeBounds = acc;
        }
    };

    for (Node* root : nodes) {
        compute(root);
    }
}

bool Model::doLoad()
{
    clear();

    const std::string basePath = AppConfig::ASSETS_PATH + "models/" + GetId();
    const std::string gltfPath = basePath + ".gltf";
    const std::string glbPath = basePath + ".glb";
    const std::string objPath = basePath + ".obj";

    // Prefer glTF/glb over OBJ.
    if (std::filesystem::exists(gltfPath)) {
        GltfModelLoader loader;
        const bool ok = loader.loadFromFile(gltfPath, *this);
        if (ok) rebuildBounds();
        return ok;
    }
    if (std::filesystem::exists(glbPath)) {
        GltfModelLoader loader;
        const bool ok = loader.loadFromFile(glbPath, *this);
        if (ok) rebuildBounds();
        return ok;
    }
    if (std::filesystem::exists(objPath)) {
        ObjModelLoader loader;
        const bool ok = loader.loadFromFile(objPath, *this);
        if (ok) rebuildBounds();
        return ok;
    }

    return false;
}

void Model::doUnload()
{
    clear();
}

