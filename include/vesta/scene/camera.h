#pragma once

#include <cstdint>

#include <glm/glm.hpp>

union SDL_Event;

// Simple fly camera used by all passes. Keeping one shared camera guarantees
// that raster, splat, and path tracing views line up exactly.
class Camera {
public:
    void SetViewport(uint32_t width, uint32_t height);
    void Focus(glm::vec3 center, float radius);
    void HandleEvent(const SDL_Event& event);
    void Update(float deltaSeconds);
    [[nodiscard]] bool ConsumeMoved();

    [[nodiscard]] glm::mat4 GetViewMatrix() const;
    [[nodiscard]] glm::mat4 GetProjectionMatrix() const;
    [[nodiscard]] glm::mat4 GetViewProjection() const;
    [[nodiscard]] glm::mat4 GetInverseViewProjection() const;
    [[nodiscard]] glm::vec3 GetPosition() const { return _position; }
    [[nodiscard]] glm::vec3 GetForward() const;
    [[nodiscard]] glm::vec3 GetUp() const { return _up; }

private:
    void UpdateDirectionFromAngles();

    glm::vec3 _position{ 0.0f, 1.5f, 5.0f };
    glm::vec3 _forward{ 0.0f, 0.0f, -1.0f };
    glm::vec3 _up{ 0.0f, 1.0f, 0.0f };
    float _yawDegrees{ -90.0f };
    float _pitchDegrees{ 0.0f };
    float _fovDegrees{ 60.0f };
    float _aspectRatio{ 16.0f / 9.0f };
    float _nearPlane{ 0.05f };
    float _farPlane{ 500.0f };
    bool _rightMouseDown{ false };
    bool _firstMouseSample{ true };
    int32_t _lastMouseX{ 0 };
    int32_t _lastMouseY{ 0 };
    bool _movedThisFrame{ true };
};
