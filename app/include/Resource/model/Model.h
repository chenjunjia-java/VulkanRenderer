#pragma once

// System
#include <memory>
#include <string>
#include <vector>

// Project
#include "Resource/core/Resource.h"
#include "Resource/model/Animation.h"
#include "Resource/model/GltfTexture.h"
#include "Resource/model/Material.h"
#include "Resource/model/Mesh.h"
#include "Resource/model/Node.h"
#include "Resource/model/Skin.h"

class Model : public Resource {
public:
    explicit Model(const std::string& id) : Resource(id) {}

    const std::vector<Node*>& getRootNodes() const { return nodes; }
    const std::vector<Node*>& getLinearNodes() const { return linearNodes; }
    const std::vector<Material>& getMaterials() const { return materials; }
    const std::vector<Mesh>& getMeshes() const { return meshes; }
    const std::vector<GltfTexture>& getTextures() const { return textures; }
    const std::vector<Skin>& getSkins() const { return skins; }
    const std::vector<Animation>& getAnimations() const { return animations; }

    Node* findNode(const std::string& name) const;
    Node* getNodeByGltfIndex(size_t index) const;
    /// Updates animation, returns true if any node transform was modified (for TLAS invalidation).
    bool updateAnimation(uint32_t index, float deltaTime);

protected:
    bool doLoad() override;
    void doUnload() override;

private:
    friend class GltfModelLoader;
    friend class ObjModelLoader;

    void clear();
    void rebuildLinearNodes();
    void rebuildBounds();

    std::vector<std::unique_ptr<Node>> ownedNodes;
    std::vector<Node*> nodes;
    std::vector<Node*> linearNodes;
    std::vector<GltfTexture> textures;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    std::vector<Skin> skins;
    std::vector<Animation> animations;
};

