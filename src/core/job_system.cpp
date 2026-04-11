#include <vesta/core/job_system.h>

#include <algorithm>

namespace vesta::core {
namespace {
thread_local bool g_isWorkerThread = false;
}

void JobSystem::Initialize(uint32_t workerCount)
{
    Shutdown();

    const uint32_t detectedConcurrency = std::max(1u, std::thread::hardware_concurrency());
    _workerCount = workerCount != 0 ? workerCount : std::max(1u, detectedConcurrency - 1u);
    _running = true;
    for (uint32_t workerIndex = 0; workerIndex < _workerCount; ++workerIndex) {
        _workers.emplace_back([this]() {
            WorkerMain();
        });
    }
}

void JobSystem::Shutdown()
{
    {
        std::scoped_lock lock(_mutex);
        _running = false;
    }
    _conditionVariable.notify_all();

    for (std::thread& worker : _workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    _workers.clear();
    _workerCount = 0;

    std::scoped_lock lock(_mutex);
    _highPriorityJobs.clear();
    _normalPriorityJobs.clear();
    _backgroundPriorityJobs.clear();
    _pendingJobCount.store(0);
}

bool JobSystem::IsWorkerThread() const
{
    return g_isWorkerThread;
}

JobHandle JobSystem::Dispatch(JobPriority priority, std::function<void()> job)
{
    auto future = Submit(priority, [task = std::move(job)]() mutable {
        task();
    });
    return JobHandle(future.share());
}

void JobSystem::Enqueue(JobPriority priority, std::function<void()> job)
{
    if (!job) {
        return;
    }

    if (_workers.empty()) {
        job();
        return;
    }

    {
        std::scoped_lock lock(_mutex);
        switch (priority) {
        case JobPriority::High:
            _highPriorityJobs.push_back(std::move(job));
            break;
        case JobPriority::Background:
            _backgroundPriorityJobs.push_back(std::move(job));
            break;
        case JobPriority::Normal:
        default:
            _normalPriorityJobs.push_back(std::move(job));
            break;
        }
        _pendingJobCount.fetch_add(1);
    }

    _conditionVariable.notify_one();
}

bool JobSystem::PopJob(std::function<void()>& job)
{
    if (!_highPriorityJobs.empty()) {
        job = std::move(_highPriorityJobs.front());
        _highPriorityJobs.pop_front();
        return true;
    }
    if (!_normalPriorityJobs.empty()) {
        job = std::move(_normalPriorityJobs.front());
        _normalPriorityJobs.pop_front();
        return true;
    }
    if (!_backgroundPriorityJobs.empty()) {
        job = std::move(_backgroundPriorityJobs.front());
        _backgroundPriorityJobs.pop_front();
        return true;
    }
    return false;
}

void JobSystem::WorkerMain()
{
    g_isWorkerThread = true;

    while (true) {
        std::function<void()> job;
        {
            std::unique_lock lock(_mutex);
            _conditionVariable.wait(lock, [this]() {
                return !_running || !_highPriorityJobs.empty() || !_normalPriorityJobs.empty() || !_backgroundPriorityJobs.empty();
            });

            if (!_running && _highPriorityJobs.empty() && _normalPriorityJobs.empty() && _backgroundPriorityJobs.empty()) {
                break;
            }

            if (!PopJob(job)) {
                continue;
            }
        }

        job();
        _pendingJobCount.fetch_sub(1);
    }

    g_isWorkerThread = false;
}
} // namespace vesta::core
