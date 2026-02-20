#include "Engine/Camera/Camera.h"

#include <algorithm>
#include <cmath>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera()
    : position(0.0f, 0.0f, 3.0f)
    , front(0.0f, 0.0f, -1.0f)
    , worldUp(0.0f, 1.0f, 0.0f)
    , yaw(-90.0f)
    , pitch(0.0f)
    , movementSpeed(2.5f)
    , mouseSensitivity(0.1f)
    , zoom(45.0f)
    , lastMouseX(0.0)
    , lastMouseY(0.0)
    , firstMouse(true)
{
    updateCameraVectors();
}

void Camera::updateCameraVectors()
{
    glm::vec3 newFront{};
    newFront.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
    newFront.y = std::sin(glm::radians(pitch));
    newFront.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
    front = glm::normalize(newFront);

    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}

void Camera::processMouseMovement(float xoffset, float yoffset)
{
    processMouseMovement(xoffset, yoffset, true);
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Clamp pitch to avoid flipping/gimbal lock.
    if (constrainPitch) {
        pitch = std::clamp(pitch, -89.0f, 89.0f);
    }

    updateCameraVectors();
}

void Camera::processMouseScroll(float yoffset)
{
    zoom -= yoffset;
    if (zoom < 1.0f) zoom = 1.0f;
    if (zoom > 45.0f) zoom = 45.0f;
}

void Camera::processMousePosition(double xpos, double ypos, bool rotate)
{
    if (!rotate) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = true;
        return;
    }

    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos - lastMouseX);
    float yoffset = static_cast<float>(lastMouseY - ypos);
    lastMouseX = xpos;
    lastMouseY = ypos;

    processMouseMovement(xoffset, yoffset);
}

void Camera::processKeyboard(float deltaTime, GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        processKeyboard(CameraMovement::FORWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        processKeyboard(CameraMovement::BACKWARD, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        processKeyboard(CameraMovement::LEFT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        processKeyboard(CameraMovement::RIGHT, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        processKeyboard(CameraMovement::DOWN, deltaTime);
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        processKeyboard(CameraMovement::UP, deltaTime);
    }
}

void Camera::processKeyboard(CameraMovement direction, float deltaTime)
{
    float velocity = movementSpeed * deltaTime;

    switch (direction) {
        case CameraMovement::FORWARD:
            position += front * velocity;
            break;
        case CameraMovement::BACKWARD:
            position -= front * velocity;
            break;
        case CameraMovement::LEFT:
            position -= right * velocity;
            break;
        case CameraMovement::RIGHT:
            position += right * velocity;
            break;
        case CameraMovement::UP:
            position += worldUp * velocity;
            break;
        case CameraMovement::DOWN:
            position -= worldUp * velocity;
            break;
    }
}

void Camera::processInput(float deltaTime, GLFWwindow* window)
{
    if (!window) return;
    processKeyboard(deltaTime, window);
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::getProjMatrix(float aspectRatio, float nearPlane, float farPlane) const
{
    glm::mat4 proj = glm::perspective(glm::radians(zoom), aspectRatio, nearPlane, farPlane);
    proj[1][1] *= -1;
    return proj;
}

Frustum Camera::GetFrustum(float aspectRatio, float nearPlane, float farPlane) const
{
    glm::mat4 view = getViewMatrix();
    glm::mat4 proj = getProjMatrix(aspectRatio, nearPlane, farPlane);
    return Frustum(proj * view);
}

