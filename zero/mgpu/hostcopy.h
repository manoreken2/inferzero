// inferzero multi-GPU: the via-host copy between two GPUs that cannot reach each other's memory (Arc B60s on
// Windows: no peer access), in pieces and on two threads. Header-only: the benchmark (bench/hostcopy_bench.cpp)
// builds it into its own program.
//
// The old path (via_host_us, in zero/mgpu_probe.cpp, which measures it) copies the whole block GPU A -> host, waits for
// A on the host, then host -> GPU B: the two links take turns, so the copy runs at half their speed. HostCopy cuts the
// block into `chunk` pieces kept in a ring of `slots` staging pieces and copies with two threads, each on its own
// command list (Level Zero: never one command list from two threads at once, ze_api.h
// zeCommandListAppendMemoryCopy), so the two directions -- two different links -- run at the same time.
//
// The pieces are handed over with a CPU flag per ring slot: a GPU-side wait on the other GPU's event is not
// something Level Zero offers across devices (ze_api.h: the events "must be accessible by the device on which the
// command list was created", and the two lists are on different devices), so each thread waits for the other with
// a spin on the flag between its own host syncs.
//
// Every wait is bounded (HostCopy::kTimeoutNs, 60 s: the sync of a piece, and the calling thread's wait for both
// halves), so a device that stops answering -- a reset, a hang -- fails the copy instead of hanging the caller;
// last_error() hands back the Level Zero result (ZE_RESULT_NOT_READY when a wait ran out). A failure inside the ring
// sticks until term() or init(): the slots may be half filled, and a device error does not heal by itself.
#pragma once
#include "../ze_min.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace hostcopy {

// The pieces on two threads, one thread per direction, with the threads started once and kept for every copy. A class,
// not a struct: the threads and the ring behind it are one object with invariants (init() / term(), one copy at a
// time), and the settings it was made with are read through piece_bytes(), ring_slots() and sync_each().
//
//   HostCopy c;
//   if (c.init(ctx, list_a, list_b) == ZE_RESULT_SUCCESS) {  // 2 MB pieces, 2 ring slots, a sync a piece (defaults)
//     const int64_t t0 = HostCopy::now_ns();               // (the clock is integer nanoseconds)
//     const ze_result_t r = c.copy(dst, src, bytes);       // one copy: pieces or the whole block, by size
//     const float us = (float)(HostCopy::now_ns() - t0) / 1000.0f;
//     // r is ZE_RESULT_SUCCESS, or the code of the call that failed (ZE_RESULT_ERROR_INVALID_ARGUMENT when this
//     // HostCopy cannot make the copy at all: not set up, empty, or too big after a ring error)
//     c.term();
//   }
//
// init() allocates pinned host memory for the ring (slots x piece: 4 MB by default) and starts the two threads; term()
// stops them and frees it. One HostCopy copies one direction pair at a time, from one thread.
class HostCopy {
 public:
  // ---- settings and constants ----------------------------------------------------------------------
  static constexpr size_t kDefaultChunk = 2 << 20;  // bytes per piece unless init() is given another

  // The smallest piece init() takes: below it the host sync of each piece costs more than the overlap saves.
  static constexpr size_t kMinChunk = 256 << 10;

  // The smallest copy copy() splits: below it two pieces overlap too briefly for the hand-over to pay, so the whole
  // block is as fast or faster.
  static constexpr size_t kMinPipelined = 384 << 10;

  // How long one wait on the device may take before the copy gives up: the command list sync of a piece, and the
  // time the calling thread waits for both halves of a round. A device that has gone (a reset, a hang, a driver
  // fault) then fails the copy (last_error() says why) instead of hanging the caller for good. Read when a copy
  // starts -- round_() copies it -- so set_timeout_ns() cannot race with a copy that is running.
  static constexpr uint64_t kTimeoutNs = 60ull * 1000000000ull;  // 60 s

  // ---- construction and lifetime -------------------------------------------------------------------
  ~HostCopy() { term(); }
  HostCopy() = default;
  HostCopy(const HostCopy&) = delete;
  HostCopy& operator=(const HostCopy&) = delete;

  // Starts the two threads and the ring. Lists: one per direction, different handles (the same from both threads is
  // what Level Zero forbids); `a` copies GPU A -> host, `b` host -> GPU B. `piece` bytes per piece (at least
  // kMinChunk: smaller pieces lose more in host syncs than the overlap wins, so init refuses them), `ring` slots
  // (pinned host memory = ring x piece; 2 is double buffering), `each` pieces per host sync (1 = sync per piece; up
  // to ring / 2). Returns ZE_RESULT_SUCCESS, ZE_RESULT_ERROR_INVALID_ARGUMENT for the arguments, or the zeMemAllocHost
  // code (which last_error() keeps too, and copy() hands back while there is nothing running).
  ze_result_t init(ze_context_handle_t context, ze_command_list_handle_t a, ze_command_list_handle_t b,
                   size_t piece = kDefaultChunk, int ring = 2, int each = 1) {
    term();
    if (!context || !a || !b || piece < kMinChunk || ring < 2 || each < 1 || each > ring / 2)
      return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    ctx_ = context;
    cl_a_ = a;
    cl_b_ = b;
    chunk_ = piece;
    slots_ = ring;
    sync_every_ = each;
    ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    const ze_result_t r = zeMemAllocHost(ctx_, &hd, (size_t)slots_ * chunk_, 64, &stage_);
    if (r != ZE_RESULT_SUCCESS) {
      stage_ = nullptr;
      ctx_ = nullptr;  // (nothing is running: no state to keep)
      cl_a_ = nullptr;
      cl_b_ = nullptr;
      rc_.store(r);
      return r;
    }
    full_ = std::vector<std::atomic<int>>((size_t)slots_);
    for (auto& f : full_) f.store(0);
    rc_.store(ZE_RESULT_SUCCESS);
    quit_.store(0);
    gen_.store(0);
    gen_start_ = gen_.load();  // (what the threads take as their first `seen`: they must not miss the first round)
    done_.store(2);  // (no round running: a copy may use the whole-block path)
    waiting_.store(0);
    running_ = true;
    prod_ = std::thread(&HostCopy::work_, this, true);
    cons_ = std::thread(&HostCopy::work_, this, false);
    return ZE_RESULT_SUCCESS;
  }

  // Stops the threads and frees the ring. Safe to call at any time, twice, or after a failed init().
  void term() {
    if (running_) {
      quit_.store(1, std::memory_order_release);
      gen_.fetch_add(1, std::memory_order_acq_rel);  // (wake them for their last look)
      {
        // Take m_ before the notify, and only then release it: a thread that has just checked its predicate under
        // m_ and has not reached cv_.wait() yet would miss a bare notify_all() and sleep for good, and join() would
        // never return. Holding the mutex closes that window -- the thread either sees quit_ in its own check or is
        // already blocked when the notify comes. (round_() takes m_ for the same reason.)
        std::lock_guard<std::mutex> lock(m_);
      }
      cv_.notify_all();
      prod_.join();
      cons_.join();
      running_ = false;
    }
    if (stage_) zeMemFree(ctx_, stage_);
    stage_ = nullptr;
    full_.clear();
    rc_.store(ZE_RESULT_SUCCESS);
    done_.store(2);  // (the two threads are gone: nothing is touching stage_ or the command lists)
    quit_.store(0);
  }

  // ---- copying -------------------------------------------------------------------------------------
  // Copies `bytes` from GPU A's memory `src` to GPU B's memory `dst`, with the faster of the two ways for that size.
  // Returns ZE_RESULT_SUCCESS, or the ze_result_t of the call that failed: a command list append / host sync inside
  // the ring (the code is kept, see last_error()) or a whole-block copy's own call. If the pieces fail and the block
  // still goes whole, the copy succeeded (the ring's error stays in last_error()). After a command list error inside
  // the ring the threads are not used again (see failed()) and this copies whole blocks, which is slower but does not
  // need them; a level zero error of that kind does not heal by itself, so a caller that wants the fast path back
  // starts a new HostCopy. ZE_RESULT_ERROR_INVALID_ARGUMENT means this HostCopy cannot make this copy at all -- not
  // initialised (after a failed init() its code is returned instead), an empty block, or (after a ring error) a block
  // larger than the whole-block path takes -- and no ze call was made for it. While a round that timed out is still
  // finishing, both halves still use the command lists and stage_, so the whole-block fallback is refused (the ring's
  // error is returned) until they come back.
  ze_result_t copy(void* dst, const void* src, size_t bytes) {
    if (!running_) {
      const ze_result_t r = last_error();
      return r != ZE_RESULT_SUCCESS ? r : ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (bytes == 0) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    const size_t piece = failed() ? 0 : piece_for(bytes);
    if (piece != 0 && copy_pipelined(dst, src, bytes, piece)) return ZE_RESULT_SUCCESS;
    const ze_result_t whole = copy_whole(dst, src, bytes);
    if (whole == ZE_RESULT_SUCCESS) return whole;
    // (both ways failed: report the ring's error -- the first failure, and the one that may be a lost device)
    const ze_result_t ring = last_error();
    return ring != ZE_RESULT_SUCCESS ? ring : whole;
  }

  // The two ways copy() chooses between. First the whole-block one, one thread: the whole block GPU A -> host, the host
  // waiting for it, then host -> GPU B and waiting again. `stage_` holds the whole block in between, so the copy must
  // fit in the ring (slots x chunk = 4 MB by default; copy() only sends it blocks under 384 KB, where the pieces cost
  // more than they save). Both waits are bounded like the ring's (timeout_ns). Returns the ze_result_t of the failing
  // call, ZE_RESULT_ERROR_INVALID_ARGUMENT when the block will not fit or the HostCopy is not running,
  // ZE_RESULT_SUCCESS when it went through. A failure here is this call's only: it does not touch last_error(), since
  // the ring is not involved. Refused (ZE_RESULT_ERROR_INVALID_ARGUMENT, or the ring's kept error) while a round is
  // still running -- after a timeout its halves have not come back -- because it would share the command lists and
  // stage_ with them.
  ze_result_t copy_whole(void* dst, const void* src, size_t bytes) {
    if (!running_) {
      const ze_result_t r = last_error();
      return r != ZE_RESULT_SUCCESS ? r : ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (done_.load(std::memory_order_acquire) < 2) {
      const ze_result_t r = last_error();
      return r != ZE_RESULT_SUCCESS ? r : ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (bytes == 0 || bytes > (size_t)slots_ * chunk_) return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    ze_result_t r = zeCommandListAppendMemoryCopy(cl_a_, stage_, src, bytes, nullptr, 0, nullptr);
    if (r != ZE_RESULT_SUCCESS) return r;
    r = zeCommandListHostSynchronize(cl_a_, timeout_ns_);
    if (r != ZE_RESULT_SUCCESS) return r;
    r = zeCommandListAppendMemoryCopy(cl_b_, dst, stage_, bytes, nullptr, 0, nullptr);
    if (r != ZE_RESULT_SUCCESS) return r;
    return zeCommandListHostSynchronize(cl_b_, timeout_ns_);  // (ZE_RESULT_SUCCESS, or the sync's code)
  }

  // The copy in pieces on two threads (at least two pieces: with fewer there is nothing to overlap). `piece` overrides
  // the piece size for this copy (the ring only has to be big enough: any piece up to `chunk` fits, including ones
  // under init()'s kMinChunk floor -- they are for measuring, not for speed); 0 = `chunk`. False if not initialised,
  // if the size holds fewer than two pieces, or on an error (which then sticks: failed()).
  bool copy_pipelined(void* dst, const void* src, size_t bytes, size_t piece = 0) {
    piece = piece != 0 ? piece : chunk_;
    if (!running_ || piece > chunk_ || bytes < piece * 2) return false;
    cur_dst_ = dst;
    cur_src_ = src;
    cur_bytes_ = bytes;
    cur_piece_ = piece;
    return round_();
  }

  // ---- state ---------------------------------------------------------------------------------------
  // The Level Zero result of the first failure in the copy path -- a command list append / host sync inside the ring,
  // or init()'s zeMemAllocHost -- (ZE_RESULT_SUCCESS while it is healthy). Kept until term() or init(): a device error
  // does not heal, and a half-finished round leaves the ring's slots in no state to continue. ZE_RESULT_NOT_READY
  // means a wait ran out (a command list sync, or round_'s wait for both halves, after timeout_ns), i.e. the device
  // stopped answering.
  ze_result_t last_error() const { return rc_.load(std::memory_order_acquire); }

  // Sticky: true once a call in the copy path (a command list append / sync inside the ring, or init()'s allocation)
  // has failed. The two threads are then left out of it until term() or init() -- copy_pipelined() returns false at
  // once -- while copy() still copies whole blocks (after a timed-out round, once its two halves have come back, since
  // they share the command lists and stage_), so a caller can tell "the fast path is gone" from "this copy failed"
  // and start a new HostCopy when it matters.
  bool failed() const { return last_error() != ZE_RESULT_SUCCESS; }

  bool ok() const { return running_; }

  // The piece size copy() uses for a copy of `bytes` -- 0 when the whole-block copy is the faster one. About bytes/8,
  // at most the ring's own piece, and never so big that the copy holds fewer than two pieces.
  size_t piece_for(size_t bytes) const {
    if (bytes < kMinPipelined) return 0;
    size_t piece = std::min(chunk_, std::max(kMinChunk, bytes / 8));
    if (piece * 2 > bytes) piece = bytes / 2;
    return piece;
  }

  // Would copy() use the pipelined path for `bytes`? (same answer as the path it takes: false once failed())
  bool pays(size_t bytes) const { return !failed() && piece_for(bytes) != 0; }

  // ---- what init() was given -----------------------------------------------------------------------
  // The settings this HostCopy was made with (for a caller that prints or logs them).
  size_t piece_bytes() const { return chunk_; }
  int ring_slots() const { return slots_; }
  int sync_each() const { return sync_every_; }

  // The wait bound a copy uses; set it between copies (a running copy keeps the value it started with). Atomic so a
  // caller that sets it while another thread copies cannot race with round_'s read.
  uint64_t timeout_ns() const { return timeout_ns_.load(std::memory_order_relaxed); }
  void set_timeout_ns(uint64_t ns) { timeout_ns_.store(ns, std::memory_order_relaxed); }

  // ---- utilities -----------------------------------------------------------------------------------
  // The clock as an integer number of nanoseconds (no double anywhere in the module): steady_clock's ticks become
  // nanoseconds exactly through duration_cast, whatever its period, and int64_t holds ~292 years of them.
  static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

 private:
  // One round: both threads copy cur_bytes_ in cur_piece_ pieces, then come back for the next round. The hand-over is
  // one atomic counter they spin on; after a while of spinning they sleep on the condition variable instead, so an
  // idle HostCopy does not hold two cores. The wait is bounded like the device waits (timeout_ns).
  bool round_() {
    cur_timeout_ns_ = timeout_ns_.load(std::memory_order_relaxed);  // (one copy's deadline: a later change cannot
                                                                    // race with a running copy)
    done_.store(0, std::memory_order_relaxed);  // (a round is running: copy_whole() stays out until both halves return)
    gen_.fetch_add(1, std::memory_order_acq_rel);
    if (waiting_.load(std::memory_order_acquire) > 0) {
      { std::lock_guard<std::mutex> lock(m_); }  // (they take m_ to sleep: wait for them to be asleep, then wake)
      cv_.notify_all();
    }
    const int64_t deadline = now_ns() + (int64_t)cur_timeout_ns_;  // (nanoseconds, integers: no rounding)
    int spins = 0;
    while (done_.load(std::memory_order_acquire) < 2) {
      YieldProcessor();
      if (++spins < 4096) continue;  // (the clock once every 4096 spins: a copy takes microseconds)
      spins = 0;
      if (now_ns() > deadline) {
        // The halves did not come back: the device is not answering (a reset, a hang). Keep a real error if one of
        // them reported one, else say the wait itself ran out -- the code a command list sync uses for that.
        ze_result_t none = ZE_RESULT_SUCCESS;
        rc_.compare_exchange_strong(none, ZE_RESULT_NOT_READY);
        return false;
      }
    }
    return rc_.load() == ZE_RESULT_SUCCESS;
  }

  // One thread's half: `producer` copies GPU A -> host into the ring, the other host -> GPU B out of it.
  void work_(bool producer) {
    const ze_command_list_handle_t cl = producer ? cl_a_ : cl_b_;
    const int want = producer ? 0 : 1;  // (a slot this thread waits for: free to fill, or filled to send out)
    unsigned seen = gen_start_;  // (the value gen_ had when these threads started: rounds increment it by one)
    for (;;) {
      // wait for the next round
      int spins = 0;
      while (gen_.load(std::memory_order_acquire) == seen) {
        if (quit_.load(std::memory_order_acquire)) return;
        if (++spins < 20000) {
          YieldProcessor();
          continue;
        }
        std::unique_lock<std::mutex> lock(m_);
        ++waiting_;
        cv_.wait(lock, [&] {
          return gen_.load(std::memory_order_acquire) != seen || quit_.load(std::memory_order_acquire);
        });
        --waiting_;
        spins = 0;
      }
      if (quit_.load(std::memory_order_acquire)) return;
      seen = gen_.load(std::memory_order_acquire);
      if (rc_.load(std::memory_order_acquire) == ZE_RESULT_SUCCESS) {
        const size_t piece = cur_piece_;
        const size_t n = (cur_bytes_ + piece - 1) / piece;
        for (size_t base = 0; base < n && rc_.load(std::memory_order_acquire) == ZE_RESULT_SUCCESS;
             base += (size_t)sync_every_) {
          const size_t g = std::min<size_t>((size_t)sync_every_, n - base);
          // every slot of this group: free (producer) / filled (consumer)
          for (size_t k = 0; k < g && rc_.load(std::memory_order_acquire) == ZE_RESULT_SUCCESS; ++k) {
            std::atomic<int>& f = full_[(base + k) % (size_t)slots_];
            while (f.load(std::memory_order_acquire) != want && rc_.load(std::memory_order_acquire) == ZE_RESULT_SUCCESS)
              YieldProcessor();
          }
          if (rc_.load(std::memory_order_acquire) != ZE_RESULT_SUCCESS) break;
          for (size_t k = 0; k < g; ++k) {
            const size_t i = base + k, off = i * piece, len = std::min(piece, cur_bytes_ - off);
            void* slot = (char*)stage_ + (i % (size_t)slots_) * chunk_;  // (slots stay `chunk` apart)
            void* d = producer ? slot : (void*)((char*)cur_dst_ + off);
            const void* s = producer ? (const void*)((const char*)cur_src_ + off) : (const void*)slot;
            const ze_result_t r = zeCommandListAppendMemoryCopy(cl, d, s, len, nullptr, 0, nullptr);
            if (r != ZE_RESULT_SUCCESS) rc_.store(r);  // (the code is kept: last_error() hands it back)
          }
          // bounded: a device that stops answering fails the copy instead of hanging this thread for good
          const ze_result_t r = zeCommandListHostSynchronize(cl, cur_timeout_ns_);
          if (r != ZE_RESULT_SUCCESS) rc_.store(r);
          for (size_t k = 0; k < g; ++k)
            full_[(base + k) % (size_t)slots_].store(producer ? 1 : 0, std::memory_order_release);
        }
      }
      done_.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  // ---- data: everything the object keeps, in one place ----------------------------------------------
  ze_context_handle_t ctx_ = nullptr;
  ze_command_list_handle_t cl_a_ = nullptr, cl_b_ = nullptr;  // one per thread
  void* stage_ = nullptr;                                     // pinned host memory: slots_ x chunk_ pieces
  size_t chunk_ = kDefaultChunk;  // bytes per piece (2 MB)
  int slots_ = 2;                 // ring depth: host memory is slots_ x chunk_. With a sync per piece the producer
                                  // cannot run more than a piece ahead, so 2 (double buffering) already keeps both
                                  // links busy; `each` pieces share one sync, so `each` <= slots_ / 2.
  int sync_every_ = 1;            // pieces per host sync per thread
  std::atomic<uint64_t> timeout_ns_{kTimeoutNs};  // (one wait's bound; see set_timeout_ns)

  std::vector<std::atomic<int>> full_;  // per slot: 1 = holds its piece (producer done), 0 = free again

  bool running_ = false;
  const void* cur_src_ = nullptr;
  void* cur_dst_ = nullptr;
  size_t cur_bytes_ = 0;
  size_t cur_piece_ = kDefaultChunk;
  uint64_t cur_timeout_ns_ = kTimeoutNs;  // (one round's copy of timeout_ns_)
  std::thread prod_, cons_;
  std::atomic<unsigned> gen_{0};    // (main: one per round; the threads wait for it to change; wraps after 4G rounds)
  unsigned gen_start_ = 0;          // (gen_ when init() started the threads: work_'s first `seen`)
  std::atomic<int> done_{2};        // (the threads: one each when their half is done; 2 = no round running)
  std::atomic<int> waiting_{0};     // (the threads asleep on the condition variable)
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<ze_result_t> rc_{ZE_RESULT_SUCCESS};  // (a ring error and its code, sticky until term())
  std::atomic<int> quit_{0};                        // (term() asks the threads to go)
};

}  // namespace hostcopy
