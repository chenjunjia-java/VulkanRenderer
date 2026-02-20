#pragma once

// System
#include <memory>
#include <string>
#include <vector>

// Project
#include "Resource/core/Resource.h"
#include "Resource/model/Material.h"
#include "Resource/model/Mesh.h"
#include "Resource/model/Node.h"

class Model : public Resource {
public:
    explicit Model(const std::string& id) : Resource(id) {}

    const std::vector<Node*>& getRootNodes() const { return nodes; }
    const std::vector<Node*>& getLinearNodes() const { return linearNodes; }
    const std::vector<Material>& getMaterials() const { return materials; }
    const std::vector<Mesh>& getMeshes() const { return meshes; }

    Node* findNode(const std::string& name) const;

protected:
    bool doLoad() override;
    void doUnload() override;

private:
    friend class GltfModelLoader;
    friend class ObjModelLoader;

    void clear();
    void rebuildLinearNodes();

    std::vector<std::unique_ptr<Node>> ownedNodes;
    std::vector<Node*> nodes;
    std::vector<Node*> linearNodes;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
};

