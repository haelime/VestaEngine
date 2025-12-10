#version 460

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform OverdrawPushConstants {
    mat4 viewProjection;
} pc;

void main()
{
    gl_Position = pc.viewProjection * vec4(inPosition, 1.0);
}
