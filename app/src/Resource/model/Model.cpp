#include "Resource/model/Model.h"

// System
#include <filesystem>

// Project
#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Resource/model/loaders/GltfModelLoader.h"
#include "Resource/model/loaders/ObjModelLoader.h"

glm::mat4 Node::getLocalMatrix() const
{
    // Match tutorial order: T * R * S * matrix
    return glm::translate(glm::mat4(1.0f), translation)
        * glm::mat4_cast(rotation)
        * glm::scale(glm::mat4(1.0f), scale)
        * matrix;
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

void Model::clear()
{
    ownedNodes.clear();
    nodes.clear();
    linearNodes.clear();
    materials.clear();
    meshes.clear();
}

void Model::rebuildLinearNodes()
{
    linearNodes.clear();
    linearNodes.reserve(ownedNodes.size());

    std::function<void(Node*)> visit = [&](Node* n) {
        if (!n) return;
        linearNodes.push_back(n);
        for (Node* c : n->children) {
            visit(c);
        }
    };

    for (Node* root : nodes) {
        visit(root);
    }
}

bool Model::doLoad()
{
    clear();

    const std::string basePath = ASSETS_PATH + "models/" + GetId();
    const std::string gltfPath = basePath + ".gltf";
    const std::string glbPath = basePath + ".glb";
    const std::string objPath = basePath + ".obj";

    // Prefer glTF/glb over OBJ.
    if (std::filesystem::exists(gltfPath)) {
        GltfModelLoader loader;
        return loader.loadFromFile(gltfPath, *this);
    }
    if (std::filesystem::exists(glbPath)) {
        GltfModelLoader loader;
        return loader.loadFromFile(glbPath, *this);
    }
    if (std::filesystem::exists(objPath)) {
        ObjModelLoader loader;
        return loader.loadFromFile(objPath, *this);
    }

    return false;
}

void Model::doUnload()
{
    clear();
}

