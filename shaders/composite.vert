#version 460

// Full-screen triangle vertex shader. Three hard-coded vertices are enough to
// cover the whole screen and avoid a dedicated full-screen vertex buffer.

vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
