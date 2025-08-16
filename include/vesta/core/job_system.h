#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace vesta::core {
enum class JobPriority : uint32_t {
    High = 0,
    Normal = 1,
    Background = 2,
};

class JobHandle {
public:
    JobHandle() = default;
    explicit JobHandle(std::shared_future<void> future)
        : _future(std::move(future))
    {
    }

    [[nodiscard]] bool valid() const { return _future.valid(); }
    void wait() const
    {
        if (_future.valid()) {
            _future.wait();
        }
    }

private:
    std::shared_future<void> _future;
};

// JobSystem is intentionally small: one shared worker pool, three priority
// queues, and futures for synchronization. The render thread stays in control
// of Vulkan while CPU-side prep work fans out here.
class JobSystem {
public:
    void Initialize(uint32_t workerCount = 0);
    void Shutdown();

    [[nodiscard]] uint32_t GetWorkerCount() const { return _workerCount; }
    [[nodiscard]] size_t GetPendingJobCount() const { return _pendingJobCount.load(); }
    [[nodiscard]] bool IsWorkerThread() const;

    JobHandle Dispatch(JobPriority priority, std::function<void()> job);

    template <typename Fn>
    auto Submit(JobPriority priority, Fn&& job) -> std::future<std::invoke_result_t<std::decay_t<Fn>>>;

    template <typename Fn>
    std::future<void> ParallelFor(size_t count, size_t grainSize, JobPriority priority, Fn&& job);

private:
    void Enqueue(JobPriority priority, std::function<void()> job);
    [[nodiscard]] bool PopJob(std::function<void()>& job);
    void WorkerMain();

    std::vector<std::thread> _workers;
    std::deque<std::function<void()>> _highPriorityJobs;
    std::deque<std::function<void()>> _normalPriorityJobs;
    std::deque<std::function<void()>> _backgroundPriorityJobs;
    std::mutex _mutex;
    std::condition_variable _conditionVariable;
    std::atomic<size_t> _pendingJobCount{ 0 };
    uint32_t _workerCount{ 0 };
    bool _running{ false };
};

template <typename Fn>
auto JobSystem::Submit(JobPriority priority, Fn&& job) -> std::future<std::invoke_result_t<std::decay_t<Fn>>>
{
    using Result = std::invoke_result_t<std::decay_t<Fn>>;

    auto promise = std::make_shared<std::promise<Result>>();
    std::future<Result> future = promise->get_future();
    auto task = std::make_shared<std::decay_t<Fn>>(std::forward<Fn>(job));

    Enqueue(priority, [promise, task]() mutable {
        try {
            if constexpr (std::is_void_v<Result>) {
                (*task)();
                promise->set_value();
            } else {
                promise->set_value((*task)());
            }
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    return future;
}

template <typename Fn>
std::future<void> JobSystem::ParallelFor(size_t count, size_t grainSize, JobPriority priority, Fn&& job)
{
    std::promise<void> emptyPromise;
    if (count == 0) {
        emptyPromise.set_value();
        return emptyPromise.get_future();
    }

    const size_t safeGrainSize = std::max<size_t>(1, grainSize);
    const size_t chunkCount = (count + safeGrainSize - 1) / safeGrainSize;
    if (_workerCount <= 1 || chunkCount <= 1) {
        return Submit(priority, [count, task = std::forward<Fn>(job)]() mutable {
            task(0, count);
        });
    }

    using Task = std::decay_t<Fn>;
    struct SharedState {
        std::promise<void> promise;
        std::shared_ptr<Task> task;
        std::atomic<size_t> remaining{ 0 };
        std::mutex exceptionMutex;
        std::exception_ptr exception;
    };

    auto state = std::make_shared<SharedState>();
    state->task = std::make_shared<Task>(std::forward<Fn>(job));
    state->remaining.store(chunkCount);
    std::future<void> future = state->promise.get_future();

    for (size_t chunkIndex = 0; chunkIndex < chunkCount; ++chunkIndex) {
        const size_t begin = chunkIndex * safeGrainSize;
        const size_t end = std::min(count, begin + safeGrainSize);
        Enqueue(priority, [state, begin, end]() mutable {
            try {
                (*state->task)(begin, end);
            } catch (...) {
                std::scoped_lock lock(state->exceptionMutex);
                if (state->exception == nullptr) {
                    state->exception = std::current_exception();
                }
            }

            if (state->remaining.fetch_sub(1) == 1) {
                if (state->exception != nullptr) {
                    state->promise.set_exception(state->exception);
                } else {
                    state->promise.set_value();
                }
            }
        });
    }

    return future;
}
} // namespace vesta::core
