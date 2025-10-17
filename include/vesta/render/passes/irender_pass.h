#pragma once

#include <string_view>

namespace vesta::render {
class RenderGraphBuilder;
class RenderGraphContext;

// 모든 렌더 패스가 따르는 공통 인터페이스다.
// Setup에서는 "무엇을 읽고/쓰는지"를 선언하고, Execute에서는 실제 명령을 기록한다.
class IRenderPass {
public:
    virtual ~IRenderPass() = default;

    // 디버그, 로그, 그래프 시각화에 쓰는 패스 이름이다.
    [[nodiscard]] virtual std::string_view Name() const = 0;
    // 이 패스가 사용하는 리소스 의존성을 그래프에 등록한다.
    virtual void Setup(RenderGraphBuilder& builder) = 0;
    // Compile된 그래프 컨텍스트를 받아 실제 Vulkan 명령을 기록한다.
    virtual void Execute(const RenderGraphContext& context) = 0;
};
} // namespace vesta::render
