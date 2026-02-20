#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Engine/Math/Frustum.h"

// Reference: LearnOpenGL camera (CN mirror): https://learnopengl-cn.github.io/01%20Getting%20started/09%20Camera/
// Y-up coordinate system. Mouse/scroll are driven by GLFW callbacks; keyboard input is polled each frame.

enum class CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Camera {
public:
    Camera();

    void processInput(float deltaTime, GLFWwindow* window);

    void processMousePosition(double xpos, double ypos, bool rotate);
    void processMouseScroll(float yoffset);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjMatrix(float aspectRatio, float nearPlane = 0.1f, float farPlane = 10.0f) const;

    Frustum GetFrustum(float aspectRatio = 16.0f / 9.0f, float nearPlane = 0.1f, float farPlane = 10.0f) const;

    glm::vec3 getPosition() const { return position; }
    glm::vec3 getFront() const { return front; }
    glm::vec3 getUp() const { return up; }

    void setPosition(const glm::vec3& pos) { position = pos; }
    void setMovementSpeed(float speed) { movementSpeed = speed; }
    void setMouseSensitivity(float sensitivity) { mouseSensitivity = sensitivity; }
    void setZoom(float z) { zoom = z; }

private:
    void updateCameraVectors();
    void processKeyboard(float deltaTime, GLFWwindow* window);
    void processKeyboard(CameraMovement direction, float deltaTime);
    void processMouseMovement(float xoffset, float yoffset);
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch);

    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;

    float yaw;
    float pitch;
    float movementSpeed;
    float mouseSensitivity;
    float zoom;

    double lastMouseX;
    double lastMouseY;
    bool firstMouse;
};

