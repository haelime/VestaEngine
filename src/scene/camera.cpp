#include <vesta/scene/camera.h>

#include <cmath>

#include <SDL.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/ext/matrix_transform.hpp>

namespace {
glm::vec3 RotateAroundAxis(const glm::vec3& vector, const glm::vec3& axis, float radians)
{
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    return vector * cosine + glm::cross(axis, vector) * sine + axis * glm::dot(axis, vector) * (1.0f - cosine);
}
} // namespace

void Camera::SetViewport(uint32_t width, uint32_t height)
{
    if (height == 0) {
        return;
    }

    _aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    _movedThisFrame = true;
}

void Camera::Focus(glm::vec3 center, float radius)
{
    _position = center + glm::vec3(radius * 0.35f, radius * 0.2f, radius * 1.4f);
    _yawDegrees = -110.0f;
    _pitchDegrees = -10.0f;
    _rollDegrees = 0.0f;
    UpdateOrientationFromAngles();
    _movedThisFrame = true;
}

void Camera::SetPosition(glm::vec3 position)
{
    const glm::vec3 delta = position - _position;
    if (glm::dot(delta, delta) <= 1.0e-10f) {
        return;
    }
    _position = position;
    _movedThisFrame = true;
}

void Camera::SetRotationDegrees(glm::vec3 rotationDegrees)
{
    rotationDegrees.y = glm::clamp(rotationDegrees.y, -89.0f, 89.0f);
    const glm::vec3 delta = rotationDegrees - glm::vec3(_yawDegrees, _pitchDegrees, _rollDegrees);
    if (glm::dot(delta, delta) <= 1.0e-10f) {
        return;
    }
    _yawDegrees = rotationDegrees.x;
    _pitchDegrees = rotationDegrees.y;
    _rollDegrees = rotationDegrees.z;
    UpdateOrientationFromAngles();
    _movedThisFrame = true;
}

void Camera::HandleEvent(const SDL_Event& event)
{
    // Right mouse button switches the camera into "mouse look" mode so the same
    // mouse can still interact with the ImGui overlay when RMB is released.
    if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_RIGHT) {
        _rightMouseDown = true;
        _firstMouseSample = true;
        SDL_SetRelativeMouseMode(SDL_TRUE);
        return;
    }

    if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_RIGHT) {
        _rightMouseDown = false;
        _firstMouseSample = true;
        SDL_SetRelativeMouseMode(SDL_FALSE);
        return;
    }

    if (event.type == SDL_MOUSEMOTION && _rightMouseDown) {
        int32_t deltaX = event.motion.xrel;
        int32_t deltaY = event.motion.yrel;
        if (_firstMouseSample) {
            deltaX = 0;
            deltaY = 0;
            _firstMouseSample = false;
        }

        constexpr float kMouseSensitivity = 0.12f;
        _yawDegrees += static_cast<float>(deltaX) * kMouseSensitivity;
        _pitchDegrees -= static_cast<float>(deltaY) * kMouseSensitivity;
        _pitchDegrees = glm::clamp(_pitchDegrees, -89.0f, 89.0f);
        UpdateOrientationFromAngles();
        _movedThisFrame = true;
    }
}

void Camera::Update(float deltaSeconds)
{
    const Uint8* keyboard = SDL_GetKeyboardState(nullptr);

    glm::vec3 right = glm::normalize(glm::cross(_forward, _up));
    glm::vec3 movement(0.0f);

    if (keyboard[SDL_SCANCODE_W]) {
        movement += _forward;
    }
    if (keyboard[SDL_SCANCODE_S]) {
        movement -= _forward;
    }
    if (keyboard[SDL_SCANCODE_D]) {
        movement += right;
    }
    if (keyboard[SDL_SCANCODE_A]) {
        movement -= right;
    }
    if (keyboard[SDL_SCANCODE_E]) {
        movement += _up;
    }
    if (keyboard[SDL_SCANCODE_Q]) {
        movement -= _up;
    }

    if (glm::dot(movement, movement) > 0.0f) {
        const bool fast = keyboard[SDL_SCANCODE_LSHIFT] != 0;
        const float speed = fast ? 8.0f : 3.0f;
        _position += glm::normalize(movement) * speed * deltaSeconds;
        _movedThisFrame = true;
    }
}

bool Camera::ConsumeMoved()
{
    const bool moved = _movedThisFrame;
    _movedThisFrame = false;
    return moved;
}

glm::mat4 Camera::GetViewMatrix() const
{
    return glm::lookAt(_position, _position + _forward, _up);
}

glm::mat4 Camera::GetProjectionMatrix() const
{
    glm::mat4 projection = glm::perspective(glm::radians(_fovDegrees), _aspectRatio, _nearPlane, _farPlane);
    // GLM follows OpenGL-style clip space by default. Flipping Y makes the
    // projection match Vulkan's screen-space convention.
    projection[1][1] *= -1.0f;
    return projection;
}

glm::mat4 Camera::GetViewProjection() const
{
    return GetProjectionMatrix() * GetViewMatrix();
}

glm::mat4 Camera::GetInverseViewProjection() const
{
    return glm::inverse(GetViewProjection());
}

glm::vec3 Camera::GetForward() const
{
    return _forward;
}

void Camera::UpdateOrientationFromAngles()
{
    const float yawRadians = glm::radians(_yawDegrees);
    const float pitchRadians = glm::radians(_pitchDegrees);

    glm::vec3 forward;
    forward.x = std::cos(yawRadians) * std::cos(pitchRadians);
    forward.y = std::sin(pitchRadians);
    forward.z = std::sin(yawRadians) * std::cos(pitchRadians);
    _forward = glm::normalize(forward);

    glm::vec3 right = glm::cross(_forward, glm::vec3(0.0f, 1.0f, 0.0f));
    if (glm::dot(right, right) <= 1.0e-8f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);
    } else {
        right = glm::normalize(right);
    }

    glm::vec3 up = glm::normalize(glm::cross(right, _forward));
    if (std::abs(_rollDegrees) > 1.0e-4f) {
        up = glm::normalize(RotateAroundAxis(up, _forward, glm::radians(_rollDegrees)));
    }
    _up = up;
}
