#version 460

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform ShadowPushConstants {
    mat4 lightViewProjection;
} pc;

void main()
{
    gl_Position = pc.lightViewProjection * vec4(inPosition, 1.0);
}
