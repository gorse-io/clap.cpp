/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <string>

int cmd_server(const std::string &host, int port, const std::string &model_path, int threads,
               const std::string &device = {}, std::size_t queue_capacity = 64);

namespace clap::server {

class Cancelled : public std::runtime_error {
  public:
    Cancelled() : std::runtime_error("Inference request cancelled") {}
};

// A request-level FIFO, not a model batcher. Only this worker may touch the
// inference handle. HTTP handlers submit owned data and wait through a Ticket.
// Capacity counts waiting requests, excluding the single running request.
template <typename Result> class Scheduler {
    using Work = std::function<Result(const std::atomic<bool> &)>;
    struct Job {
        explicit Job(Work work) : work(std::move(work)), result(promise.get_future()) {}
        Work work;
        std::promise<Result> promise;
        std::future<Result> result;
        std::atomic<bool> cancelled{false};
    };
    struct State {
        std::mutex mutex;
        std::condition_variable ready;
        std::deque<std::shared_ptr<Job>> queue;
        std::shared_ptr<Job> active;
        bool stopping = false;
    };

    static void cancel(const std::shared_ptr<State> &state, const std::shared_ptr<Job> &job) {
        if (!state || !job)
            return;
        const std::lock_guard<std::mutex> lock(state->mutex);
        job->cancelled = true;
        const auto found = std::find(state->queue.begin(), state->queue.end(), job);
        if (found != state->queue.end()) {
            job->promise.set_exception(std::make_exception_ptr(Cancelled()));
            state->queue.erase(found);
        }
        // Running work owns its completion; never set its promise here.
    }

  public:
    class Ticket {
      public:
        Ticket() = default;
        Ticket(const Ticket &) = delete;
        Ticket &operator=(const Ticket &) = delete;
        Ticket(Ticket &&other) noexcept : state_(std::move(other.state_)), job_(std::move(other.job_)) {}
        Ticket &operator=(Ticket &&other) noexcept {
            if (this != &other) {
                cancel();
                state_ = std::move(other.state_);
                job_ = std::move(other.job_);
            }
            return *this;
        }
        ~Ticket() { cancel(); }
        explicit operator bool() const { return bool(job_); }
        std::future<Result> &future() { return job_->result; }
        void cancel() { Scheduler::cancel(state_.lock(), job_); }

      private:
        friend class Scheduler;
        Ticket(const std::shared_ptr<State> &state, std::shared_ptr<Job> job) : state_(state), job_(std::move(job)) {}
        std::weak_ptr<State> state_;
        std::shared_ptr<Job> job_;
    };

    // Initialization, every job, and finalization execute on the same thread.
    // Construction waits for initialization (or forwards its exception).
    // finalize must not throw, including after partially failed initialization.
    explicit Scheduler(std::size_t capacity, std::function<void()> initialize = {}, std::function<void()> finalize = {})
        : capacity_(capacity) {
        if (!capacity)
            throw std::invalid_argument("Queue capacity must be positive");
        std::promise<void> started;
        auto startup = started.get_future();
        worker_ = std::thread([state = state_, initialize = std::move(initialize), finalize = std::move(finalize),
                               started = std::move(started)]() mutable {
            try {
                if (initialize)
                    initialize();
            } catch (...) {
                const auto error = std::current_exception();
                if (finalize)
                    finalize();
                started.set_exception(error);
                return;
            }
            started.set_value();
            for (;;) {
                std::shared_ptr<Job> job;
                {
                    std::unique_lock<std::mutex> lock(state->mutex);
                    state->ready.wait(lock, [&] { return state->stopping || !state->queue.empty(); });
                    if (state->stopping)
                        break;
                    job = state->queue.front();
                    state->queue.pop_front();
                    state->active = job;
                }
                try {
                    if (job->cancelled)
                        throw Cancelled();
                    auto result = job->work(job->cancelled);
                    const std::lock_guard<std::mutex> lock(state->mutex);
                    if (job->cancelled)
                        job->promise.set_exception(std::make_exception_ptr(Cancelled()));
                    else
                        job->promise.set_value(std::move(result));
                    state->active.reset();
                } catch (...) {
                    const auto error = std::current_exception();
                    const std::lock_guard<std::mutex> lock(state->mutex);
                    job->promise.set_exception(error);
                    state->active.reset();
                }
            }
            if (finalize)
                finalize();
        });
        try {
            startup.get();
        } catch (...) {
            worker_.join();
            throw;
        }
    }

    Scheduler(const Scheduler &) = delete;
    Scheduler &operator=(const Scheduler &) = delete;
    ~Scheduler() { stop(); }

    Ticket submit(Work work) {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->stopping || state_->queue.size() >= capacity_)
            return {};
        auto job = std::make_shared<Job>(std::move(work));
        state_->queue.push_back(job);
        state_->ready.notify_one();
        return Ticket(state_, std::move(job));
    }

    // Reject new work, resolve all queued waiters, cooperatively cancel the
    // current request, then join before destroying model/runtime state. A C API
    // call already in progress cannot be interrupted; it is allowed to finish.
    // Call from outside the worker; concurrent external stop calls are safe.
    void stop() {
        const std::lock_guard<std::mutex> joining(join_mutex_);
        {
            const std::lock_guard<std::mutex> lock(state_->mutex);
            state_->stopping = true;
            for (const auto &job : state_->queue) {
                job->cancelled = true;
                job->promise.set_exception(std::make_exception_ptr(Cancelled()));
            }
            state_->queue.clear();
            if (state_->active)
                state_->active->cancelled = true;
        }
        state_->ready.notify_all();
        if (worker_.joinable())
            worker_.join();
    }

  private:
    const std::size_t capacity_;
    const std::shared_ptr<State> state_ = std::make_shared<State>();
    std::mutex join_mutex_;
    std::thread worker_;
};

} // namespace clap::server
