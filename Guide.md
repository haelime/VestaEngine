🏗️ Vulkan 엔진 아키텍처 설계 가이드
핵심 원칙
1. RAII: 생성자에서 할당, 소멸자에서 해제
2. 소유권 명확화: 누가 뭘 소유하는지 타입으로 표현
3. 불변 의존성: 객체 생성 시 필요한 것들은 생성자로 주입
4. 계층 분리: Low-level Vulkan 래퍼 ↔ High-level 엔진 로직
---
계층 구조
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
│         (Game Logic, Scene, Entity 등)                          │
├─────────────────────────────────────────────────────────────────┤
│                    Engine Layer                                  │
│         (Renderer, ResourceManager, RenderGraph 등)             │
├─────────────────────────────────────────────────────────────────┤
│                    RHI (Render Hardware Interface)               │
│         (Vulkan 추상화 - 나중에 다른 API 지원 가능)              │
├─────────────────────────────────────────────────────────────────┤
│                    Vulkan Wrapper Layer                          │
│         (RAII 래퍼 - VkImage → vesta::Image 등)                 │
├─────────────────────────────────────────────────────────────────┤
│                    Raw Vulkan API                                │
└─────────────────────────────────────────────────────────────────┘
---
Layer 1: Vulkan RAII Wrapper
설계 패턴: Handle Wrapper
// 기본 구조
namespace vesta::vk {
// Non-copyable, Movable
class Image {
public:
    Image(Device& device, const ImageCreateInfo& info);
    ~Image();
    
    // Move only
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    
    // Raw handle 접근 (필요할 때만)
    VkImage handle() const { return image_; }
    
private:
    Device* device_;  // non-owning reference
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;  // 또는 별도 Allocation 객체
};
}
핵심 래퍼 클래스 목록
┌──────────────────────────────────────────────────────────────────┐
│                     Core Wrappers                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Instance          - VkInstance + Debug Messenger                │
│       │                                                          │
│       └──→ Device  - VkDevice + VkPhysicalDevice + Queues       │
│              │                                                   │
│              ├──→ Buffer          - VkBuffer + Memory            │
│              ├──→ Image           - VkImage + Memory + View      │
│              ├──→ Sampler         - VkSampler                    │
│              ├──→ ShaderModule    - VkShaderModule               │
│              ├──→ PipelineLayout  - VkPipelineLayout             │
│              ├──→ Pipeline        - VkPipeline                   │
│              ├──→ RenderPass      - VkRenderPass                 │
│              ├──→ Framebuffer     - VkFramebuffer                │
│              ├──→ CommandPool     - VkCommandPool                │
│              ├──→ CommandBuffer   - VkCommandBuffer (풀에서 할당)│
│              ├──→ DescriptorPool  - VkDescriptorPool             │
│              ├──→ DescriptorSet   - VkDescriptorSet              │
│              ├──→ Fence           - VkFence                      │
│              └──→ Semaphore       - VkSemaphore                  │
│                                                                   │
│  Surface           - VkSurfaceKHR (Window와 연결)                │
│  SwapChain         - VkSwapchainKHR + Images                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
---
소유권 모델
패턴 1: Unique Ownership (대부분의 경우)
class Device {
    // Device가 소유하는 것들
    VkDevice device_;
    VkPhysicalDevice physicalDevice_;  // 소유 아님 (Instance가 관리)
    
    // Queues - Device 생성 시 함께 생성됨
    VkQueue graphicsQueue_;
    VkQueue presentQueue_;
};
패턴 2: Non-owning Reference
class Buffer {
    Device* device_;  // 포인터 = non-owning
    // Buffer는 Device보다 먼저 소멸되어야 함
};
class CommandBuffer {
    CommandPool* pool_;  // Pool에서 할당받음
    // 개별 해제 or Pool 리셋으로 일괄 해제
};
패턴 3: Shared Ownership (드물게)
// Pipeline이 여러 RenderPass에서 호환 가능할 때
class Pipeline {
    std::shared_ptr<PipelineLayout> layout_;
    // 여러 Pipeline이 같은 Layout 공유 가능
};
---
생성자 의존성 주입
Create Info 구조체 패턴
namespace vesta::vk {
struct ImageCreateInfo {
    VkExtent3D extent;
    VkFormat format;
    VkImageUsageFlags usage;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    
    // Builder 패턴도 고려
    ImageCreateInfo& setExtent(uint32_t w, uint32_t h, uint32_t d = 1);
    ImageCreateInfo& setFormat(VkFormat fmt);
    // ...
};
class Image {
public:
    Image(Device& device, const ImageCreateInfo& info);
};
}
사용 예시
auto image = vesta::vk::Image(device, 
    ImageCreateInfo{}
        .setExtent(800, 600)
        .setFormat(VK_FORMAT_R8G8B8A8_SRGB)
        .setUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
);
---
메모리 관리 전략
옵션 1: 단순 (학습용)
class Buffer {
    VkBuffer buffer_;
    VkDeviceMemory memory_;  // Buffer마다 개별 할당
};
옵션 2: Memory Allocator 분리 (실제 엔진)
class Allocator {
public:
    Allocation allocate(const AllocationRequirements& req);
    void free(Allocation& alloc);
};
class Buffer {
    VkBuffer buffer_;
    Allocation allocation_;  // Allocator가 관리하는 메모리 조각
};
추천: VMA (Vulkan Memory Allocator) 사용
// AMD의 VMA 라이브러리 - 프로덕션 레벨
class Allocator {
    VmaAllocator allocator_;
public:
    // VMA가 알아서 메모리 풀링, 조각화 방지 처리
};
---
Layer 2: High-Level Abstractions
SwapChain + Frame 관리
class Swapchain {
public:
    Swapchain(Device& device, Surface& surface, const SwapchainConfig& config);
    
    // 재생성 지원 (창 리사이즈)
    void recreate(uint32_t width, uint32_t height);
    
    // Frame 획득
    struct AcquireResult {
        uint32_t imageIndex;
        bool needsRecreate;  // VK_SUBOPTIMAL_KHR 등
    };
    AcquireResult acquireNextImage(Semaphore& signalSemaphore);
    
    // Present
    void present(uint32_t imageIndex, Semaphore& waitSemaphore);
    
private:
    std::vector<Image> images_;  // Swapchain이 소유 (자동 생성됨)
    std::vector<ImageView> imageViews_;
};
Frame In Flight 추상화
class FrameContext {
public:
    // 현재 프레임의 리소스들
    CommandBuffer& commandBuffer();
    Semaphore& imageAvailableSemaphore();
    Semaphore& renderFinishedSemaphore();
    Fence& inFlightFence();
    
    // 프레임별 동적 데이터 (매 프레임 리셋)
    // - Staging buffer
    // - Descriptor 업데이트
};
class FrameManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    
    FrameContext& beginFrame();  // Fence 대기, 리소스 준비
    void endFrame();             // Submit + Present
    
private:
    std::array<FrameContext, MAX_FRAMES_IN_FLIGHT> frames_;
    uint32_t currentFrame_ = 0;
};
---
Layer 3: Render Abstraction
RenderPass + Framebuffer 추상화
struct AttachmentDescription {
    VkFormat format;
    VkAttachmentLoadOp loadOp;
    VkAttachmentStoreOp storeOp;
    VkImageLayout initialLayout;
    VkImageLayout finalLayout;
};
class RenderPassBuilder {
public:
    RenderPassBuilder& addColorAttachment(const AttachmentDescription& desc);
    RenderPassBuilder& setDepthAttachment(const AttachmentDescription& desc);
    RenderPassBuilder& addSubpass(/* ... */);
    
    RenderPass build(Device& device);
};
Pipeline 추상화
class GraphicsPipelineBuilder {
public:
    GraphicsPipelineBuilder& setShaders(ShaderModule& vert, ShaderModule& frag);
    GraphicsPipelineBuilder& setVertexInput(const VertexInputDescription& desc);
    GraphicsPipelineBuilder& setInputAssembly(VkPrimitiveTopology topology);
    GraphicsPipelineBuilder& setRasterization(const RasterizationState& state);
    GraphicsPipelineBuilder& setMultisample(const MultisampleState& state);
    GraphicsPipelineBuilder& setDepthStencil(const DepthStencilState& state);
    GraphicsPipelineBuilder& setColorBlend(const ColorBlendState& state);
    GraphicsPipelineBuilder& setDynamicStates(std::span<VkDynamicState> states);
    GraphicsPipelineBuilder& setLayout(PipelineLayout& layout);
    GraphicsPipelineBuilder& setRenderPass(RenderPass& pass, uint32_t subpass = 0);
    
    Pipeline build(Device& device);
};
---
파일/폴더 구조 제안
VestaEngine/
├── src/
│   ├── core/                    # 기본 유틸리티
│   │   ├── Types.hpp            # 기본 타입 정의
│   │   └── Result.hpp           # 에러 처리 (std::expected 등)
│   │
│   ├── vk/                      # Vulkan RAII Wrappers
│   │   ├── Instance.hpp/cpp
│   │   ├── Device.hpp/cpp
│   │   ├── Buffer.hpp/cpp
│   │   ├── Image.hpp/cpp
│   │   ├── ImageView.hpp/cpp
│   │   ├── Sampler.hpp/cpp
│   │   ├── ShaderModule.hpp/cpp
│   │   ├── Pipeline.hpp/cpp
│   │   ├── PipelineLayout.hpp/cpp
│   │   ├── RenderPass.hpp/cpp
│   │   ├── Framebuffer.hpp/cpp
│   │   ├── CommandPool.hpp/cpp
│   │   ├── CommandBuffer.hpp/cpp
│   │   ├── DescriptorPool.hpp/cpp
│   │   ├── DescriptorSet.hpp/cpp
│   │   ├── Fence.hpp/cpp
│   │   ├── Semaphore.hpp/cpp
│   │   ├── Surface.hpp/cpp
│   │   ├── Swapchain.hpp/cpp
│   │   └── Allocator.hpp/cpp    # 메모리 할당
│   │
│   ├── renderer/                # High-level 렌더링
│   │   ├── FrameManager.hpp/cpp
│   │   ├── RenderPassBuilder.hpp/cpp
│   │   ├── PipelineBuilder.hpp/cpp
│   │   └── Renderer.hpp/cpp
│   │
│   ├── window/                  # 윈도우 시스템
│   │   └── Window.hpp/cpp       # GLFW 래핑
│   │
│   └── app/                     # 애플리케이션
│       └── Application.hpp/cpp
---
구현 순서 제안
Phase 1: Core Wrappers
1. Instance (+ Debug Messenger)
2. Device (+ Queue 획득)
3. Surface
4. Swapchain (기본)
5. Fence, Semaphore
6. CommandPool, CommandBuffer
Phase 2: Resource Wrappers
7. Buffer (+ 단순 메모리 할당)
8. Image, ImageView
9. ShaderModule
10. RenderPass
11. Framebuffer
12. PipelineLayout, Pipeline
Phase 3: High-level Abstractions
13. FrameManager (Frame In Flight)
14. Swapchain 재생성
15. Pipeline Builder
16. Descriptor 시스템
Phase 4: Advanced
17. VMA 통합
18. RenderGraph (선택)
19. 멀티스레드 커맨드 기록
---
에러 처리 전략
옵션 1: 예외 (현재 VestaEngine 방식)
if (vkCreateBuffer(...) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer");
}
옵션 2: std::expected (C++23) 또는 유사 타입
template<typename T>
using Result = std::expected<T, VkResult>;
Result<Buffer> Buffer::create(Device& device, const BufferCreateInfo& info) {
    VkBuffer buffer;
    VkResult result = vkCreateBuffer(...);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }
    return Buffer(device, buffer, ...);
}
// 사용
auto bufferResult = Buffer::create(device, info);
if (!bufferResult) {
    // 에러 처리
    log::error("Buffer creation failed: {}", bufferResult.error());
    return;
}
auto buffer = std::move(*bufferResult);
옵션 3: 디버그 모드에서만 체크
#ifdef VESTA_DEBUG
    #define VK_CHECK(result) \
        do { \
            if ((result) != VK_SUCCESS) { \
                throw std::runtime_error("Vulkan error: " #result); \
            } \
        } while(0)
#else
    #define VK_CHECK(result) (void)(result)
#endif
---
추가 고려사항
1. Deletion Queue 패턴
프레임이 완료될 때까지 리소스 삭제 지연:
class DeletionQueue {
public:
    void push(std::function<void()>&& deletor);
    void flush();  // 모든 deletor 실행
    
private:
    std::vector<std::function<void()>> deletors_;
};
// 사용
deletionQueue.push([=]() {
    vkDestroyBuffer(device, buffer, nullptr);
});
2. Handle Validation (디버그)
class Buffer {
#ifdef VESTA_DEBUG
    bool isValid() const { return buffer_ != VK_NULL_HANDLE; }
#endif
public:
    VkBuffer handle() const { 
        assert(isValid() && "Accessing invalid buffer");
        return buffer_; 
    }
};
3. 스레드 안전성
class CommandPool {
    // Pool은 스레드당 하나씩 사용
    // 또는 mutex로 보호
};
class DescriptorPool {
    std::mutex mutex_;  // Allocate/Free 시 락
};
---
요약
| 원칙 | 적용 |
|------|------|
| RAII | 모든 Vulkan 핸들을 클래스로 래핑 |
| Move-only | 복사 금지, 이동만 허용 |
| 의존성 주입 | 필요한 객체는 생성자로 전달 |
| Builder 패턴 | 복잡한 객체 생성 시 |
| 계층 분리 | Raw Vulkan → Wrapper → High-level |
시작점으로 Instance → Device → Swapchain 순서로 래핑하면서 패턴을 익히는 걸 추천합니다.