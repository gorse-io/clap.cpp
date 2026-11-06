#include "server.h"

#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using Scheduler = clap::server::Scheduler<int>;

void require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}

void cancelled(Scheduler::Ticket &ticket) {
    try {
        (void)ticket.future().get();
        throw std::runtime_error("cancelled job returned a result");
    } catch (const clap::server::Cancelled &) {
    }
}

// A deterministic barrier: no timing guesses or sleeps in scheduling tests.
struct Gate {
    std::promise<void> entered;
    std::promise<void> release;
    std::shared_future<void> released = release.get_future().share();
    int run(const std::atomic<bool> &) {
        entered.set_value();
        released.wait();
        return 1;
    }
    void wait() { entered.get_future().wait(); }
    void open() { release.set_value(); }
};

void fifo_capacity_atomicity() {
    Scheduler scheduler(2);
    Gate gate;
    auto active = scheduler.submit([&](const auto &cancel) { return gate.run(cancel); });
    gate.wait();
    std::vector<int> order;
    auto first = scheduler.submit([&](const auto &) {
        // A whole request, not independently scheduled embedding inputs.
        order.push_back(2);
        order.push_back(3);
        return 23;
    });
    auto second = scheduler.submit([&](const auto &) {
        order.push_back(4);
        return 4;
    });
    auto rejected = scheduler.submit([](const auto &) { return 0; });
    require(!rejected, "full queue must reject immediately");
    gate.open();
    require(active.future().get() == 1, "active result");
    require(first.future().get() == 23, "request result");
    require(second.future().get() == 4, "FIFO result");
    require(order == std::vector<int>({2, 3, 4}), "FIFO and atomic request execution");
}

void queued_cancellation_releases_capacity() {
    Scheduler scheduler(1);
    Gate gate;
    auto active = scheduler.submit([&](const auto &cancel) { return gate.run(cancel); });
    gate.wait();
    std::atomic<bool> ran{false};
    auto queued = scheduler.submit([&](const auto &) {
        ran = true;
        return 0;
    });
    queued.cancel();
    queued.cancel(); // idempotent: no double completion
    cancelled(queued);
    auto replacement = scheduler.submit([](const auto &) { return 5; });
    require(bool(replacement), "cancellation must remove queued job immediately");
    gate.open();
    active.future().get();
    require(replacement.future().get() == 5 && !ran, "cancelled job must not run");
}

void abandoned_waiter_cancels() {
    Scheduler scheduler(1);
    Gate gate;
    auto active = scheduler.submit([&](const auto &cancel) { return gate.run(cancel); });
    gate.wait();
    std::atomic<bool> ran{false};
    {
        auto abandoned = scheduler.submit([&](const auto &) {
            ran = true;
            return 0;
        });
        require(bool(abandoned), "accepted waiter");
    }
    auto next = scheduler.submit([](const auto &) { return 7; });
    require(bool(next), "abandoned waiter frees capacity");
    gate.open();
    active.future().get();
    require(next.future().get() == 7 && !ran, "abandoned waiter never executes");
}

void active_cancellation_is_cooperative() {
    Scheduler scheduler(1);
    Gate gate;
    std::atomic<int> inputs{0};
    auto active = scheduler.submit([&](const std::atomic<bool> &cancel) {
        ++inputs;
        gate.run(cancel);
        if (!cancel.load())
            ++inputs;
        return 2;
    });
    gate.wait();
    active.cancel();
    gate.open();
    cancelled(active);
    require(inputs == 1, "cancel before next input");
    auto next = scheduler.submit([](const auto &) { return 8; });
    require(next.future().get() == 8, "worker survives cancellation");
}

void failure_does_not_poison_worker() {
    Scheduler scheduler(1);
    auto failed = scheduler.submit([](const auto &) -> int { throw std::logic_error("inference failed"); });
    try {
        failed.future().get();
        throw std::runtime_error("missing inference failure");
    } catch (const std::logic_error &error) {
        require(std::string(error.what()) == "inference failed", "preserved error");
    }
    auto next = scheduler.submit([](const auto &) { return 9; });
    require(next.future().get() == 9, "worker survives failure");
}

void orderly_stop_and_owner_lifetime() {
    std::thread::id owner;
    bool destroyed = false;
    Scheduler scheduler(
        1, [&] { owner = std::this_thread::get_id(); },
        [&] {
            require(owner == std::this_thread::get_id(), "destroy on owner thread");
            destroyed = true;
        });
    Gate gate;
    auto active = scheduler.submit([&](const auto &cancel) {
        require(owner == std::this_thread::get_id(), "infer on owner thread");
        return gate.run(cancel);
    });
    gate.wait();
    auto queued = scheduler.submit([](const auto &) { return 0; });
    auto stopper = std::async(std::launch::async, [&] { scheduler.stop(); });
    // stop resolves queued work without waiting for the in-flight C call.
    cancelled(queued);
    require(!scheduler.submit([](const auto &) { return 0; }), "stop rejects submissions");
    gate.open();
    cancelled(active);
    stopper.get();
    scheduler.stop();
    require(destroyed, "stop joins worker and destroys owner");
}

void destructor_joins_and_startup_failure_propagates() {
    bool finalized = false;
    {
        Scheduler scheduler(1, [] {}, [&] { finalized = true; });
        auto job = scheduler.submit([](const auto &) { return 1; });
        require(job.future().get() == 1, "completed before destruction");
    }
    require(finalized, "destructor joins");
    try {
        Scheduler scheduler(1, [] { throw std::runtime_error("load failed"); });
        throw std::logic_error("startup did not fail");
    } catch (const std::runtime_error &error) {
        require(std::string(error.what()) == "load failed", "startup error forwarded");
    }
    try {
        Scheduler scheduler(0);
        throw std::logic_error("zero capacity accepted");
    } catch (const std::invalid_argument &) {
    }
}

int main() {
    try {
        fifo_capacity_atomicity();
        queued_cancellation_releases_capacity();
        abandoned_waiter_cancels();
        active_cancellation_is_cooperative();
        failure_does_not_poison_worker();
        orderly_stop_and_owner_lifetime();
        destructor_joins_and_startup_failure_propagates();
        std::cout << "7 scheduler tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
