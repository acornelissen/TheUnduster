//! Per-roll background job queue state.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Detect,
    Heal,
    Export,
    /// Warms a neighbor frame into the registry (decode + pyramid) without
    /// detecting or healing it, so stepping through a roll hits a resident
    /// entry instead of paying the full decode on arrival.
    Prefetch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Job {
    pub kind: JobKind,
    pub index: usize,
    /// The roll generation this job was enqueued against. Included in the
    /// derived equality (and therefore in coalescing): the same frame index
    /// in two different rolls is a different job, so a stale-roll job must
    /// never coalesce with -- or be mistaken for -- a same-index job queued
    /// after a roll swap.
    pub generation: u64,
}

/// Per-roll job queue. One worker drains it (single-flight `running` flag,
/// drop-guard cleared); `pinned` is the image id of the frame the active
/// job is processing, which eviction must skip.
#[derive(Default)]
pub struct JobQueue {
    queue: Mutex<VecDeque<Job>>,
    pub running: AtomicBool,
    pinned: Mutex<Option<u64>>,
    /// The job the worker is currently executing, if any. Both the worker
    /// (begin_job/end_job) and cancel requests (request_cancel*) take this
    /// lock, so "does this cancel target the job that is running right now"
    /// is decided under one guard -- a cancel aimed at a job that just
    /// finished can never leak onto the next job the worker picks up.
    running_job: Mutex<Option<Job>>,
    /// Cooperative abort flag for the running job. Set only under the
    /// `running_job` lock (see request_cancel), cleared by begin_job before
    /// each job starts; job bodies read it lock-free from their progress
    /// callbacks and between export stages.
    cancel_requested: AtomicBool,
}

impl JobQueue {
    /// Enqueues a job, with optional front-priority insertion. Returns true if
    /// the job was newly enqueued, false if an equal job was already queued
    /// (coalesced). When `front` is true, an existing equal job is moved to
    /// the front rather than duplicated.
    pub fn enqueue(&self, job: Job, front: bool) -> Result<bool, String> {
        let mut queue = self.queue.lock().map_err(|e| e.to_string())?;

        // Check if an equal job already exists
        if let Some(pos) = queue.iter().position(|&j| j == job) {
            if front {
                // Remove the existing equal job and push to front
                queue.remove(pos);
                queue.push_front(job);
            }
            // Return false: coalesced (not newly added)
            return Ok(false);
        }

        // No equal job exists; insert the new one
        if front {
            queue.push_front(job);
        } else {
            queue.push_back(job);
        }

        Ok(true)
    }

    /// Removes and returns the front job from the queue, or None if empty.
    pub fn pop(&self) -> Result<Option<Job>, String> {
        let mut queue = self.queue.lock().map_err(|e| e.to_string())?;
        Ok(queue.pop_front())
    }

    /// Removes all jobs from the queue and returns how many were dropped.
    pub fn clear(&self) -> Result<usize, String> {
        let mut queue = self.queue.lock().map_err(|e| e.to_string())?;
        let count = queue.len();
        queue.clear();
        Ok(count)
    }

    /// Removes the job equal to `job` (kind, index AND generation must all
    /// match) from the pending queue. Returns true if it was there. The
    /// running job is not in the queue, so this never touches it -- that is
    /// `request_cancel`'s side of the split.
    pub fn remove(&self, job: Job) -> Result<bool, String> {
        let mut queue = self.queue.lock().map_err(|e| e.to_string())?;
        match queue.iter().position(|&j| j == job) {
            Some(pos) => {
                queue.remove(pos);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Empties the queue and returns the removed jobs in order (front
    /// first), so cancel-all can emit a job-cancelled event per dropped job
    /// -- `clear` only counts them.
    pub fn drain(&self) -> Result<Vec<Job>, String> {
        let mut queue = self.queue.lock().map_err(|e| e.to_string())?;
        Ok(queue.drain(..).collect())
    }

    /// Records `job` as the one the worker is executing and clears any
    /// cancel request left over from the previous job, so a cancel that
    /// arrived too late to stop its target can never abort an innocent
    /// successor.
    pub fn begin_job(&self, job: Job) {
        let mut running = self.running_job.lock().unwrap_or_else(|e| e.into_inner());
        self.cancel_requested.store(false, Ordering::SeqCst);
        *running = Some(job);
    }

    /// Clears the running-job record (and the cancel flag, for symmetry)
    /// once the worker is done with it.
    pub fn end_job(&self) {
        let mut running = self.running_job.lock().unwrap_or_else(|e| e.into_inner());
        self.cancel_requested.store(false, Ordering::SeqCst);
        *running = None;
    }

    /// Requests a cooperative abort of `job` IF it is the one currently
    /// running. Returns true when the request landed. Decided under the
    /// running_job lock -- see that field's doc comment for the race this
    /// closes.
    pub fn request_cancel(&self, job: Job) -> bool {
        let running = self.running_job.lock().unwrap_or_else(|e| e.into_inner());
        if *running == Some(job) {
            self.cancel_requested.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Requests a cooperative abort of whatever job is running, if any --
    /// cancel-all's counterpart to `request_cancel`.
    pub fn request_cancel_running(&self) -> bool {
        let running = self.running_job.lock().unwrap_or_else(|e| e.into_inner());
        if running.is_some() {
            self.cancel_requested.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// True when the running job has been asked to stop. Job bodies poll
    /// this from progress callbacks and between export stages.
    pub fn cancel_requested(&self) -> bool {
        self.cancel_requested.load(Ordering::SeqCst)
    }

    /// Returns the number of jobs currently in the queue.
    // Exercised by tests; no production caller yet.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn len(&self) -> Result<usize, String> {
        let queue = self.queue.lock().map_err(|e| e.to_string())?;
        Ok(queue.len())
    }

    /// Clones the pending queue in order (front first), leaving it intact.
    /// Backs the `queue_snapshot` command so the UI can show pending jobs
    /// without draining them.
    pub fn snapshot(&self) -> Result<Vec<Job>, String> {
        let queue = self.queue.lock().map_err(|e| e.to_string())?;
        Ok(queue.iter().copied().collect())
    }

    /// Returns true if the queue is empty. The worker's clear-then-recheck
    /// exit handshake depends on this re-check happening after the running
    /// flag clears.
    pub fn is_empty(&self) -> Result<bool, String> {
        let queue = self.queue.lock().map_err(|e| e.to_string())?;
        Ok(queue.is_empty())
    }

    /// Stores the id of the frame the active job is processing (or None to clear).
    pub fn pin(&self, id: Option<u64>) -> Result<(), String> {
        let mut pinned = self.pinned.lock().map_err(|e| e.to_string())?;
        *pinned = id;
        Ok(())
    }

    /// Retrieves the currently pinned frame id, if any.
    pub fn pinned(&self) -> Result<Option<u64>, String> {
        let pinned = self.pinned.lock().map_err(|e| e.to_string())?;
        Ok(*pinned)
    }

    /// Clears the running flag. Called by the drop guard when a job-processing
    /// task completes or unwinds, mirroring RollState::clear_scanning.
    pub fn clear_running(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Attempts to claim the running flag (false -> true). Returns true on
    /// success. Shared by `enqueue_job`'s initial claim and the worker's
    /// clear-then-recheck exit handshake, so both sites use the identical
    /// compare_exchange ordering.
    pub fn try_start(&self) -> bool {
        self.running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(kind: JobKind, index: usize) -> Job {
        job_gen(kind, index, 0)
    }

    fn job_gen(kind: JobKind, index: usize, generation: u64) -> Job {
        Job {
            kind,
            index,
            generation,
        }
    }

    #[test]
    fn enqueue_coalesces_and_prioritizes() {
        let q = JobQueue::default();
        assert!(q.enqueue(job(JobKind::Heal, 1), false).unwrap());
        assert!(q.enqueue(job(JobKind::Detect, 1), false).unwrap()); // different kind, kept
        assert!(!q.enqueue(job(JobKind::Heal, 1), false).unwrap()); // duplicate coalesced
        assert_eq!(q.len().unwrap(), 2);
        // front insertion moves an existing equal job rather than duplicating
        assert!(!q.enqueue(job(JobKind::Detect, 1), true).unwrap());
        assert_eq!(q.len().unwrap(), 2);
        assert_eq!(q.pop().unwrap(), Some(job(JobKind::Detect, 1)));
        assert_eq!(q.pop().unwrap(), Some(job(JobKind::Heal, 1)));
        assert_eq!(q.pop().unwrap(), None);
    }

    #[test]
    fn jobs_differing_only_by_generation_do_not_coalesce() {
        let q = JobQueue::default();
        assert!(q.enqueue(job_gen(JobKind::Heal, 1, 0), false).unwrap());
        // Same kind and index, but a different roll generation: this is a
        // distinct job (a straggler from the old roll must not coalesce with
        // -- or be silently replaced by -- the new roll's job).
        assert!(q.enqueue(job_gen(JobKind::Heal, 1, 1), false).unwrap());
        assert_eq!(q.len().unwrap(), 2);
        assert_eq!(q.pop().unwrap(), Some(job_gen(JobKind::Heal, 1, 0)));
        assert_eq!(q.pop().unwrap(), Some(job_gen(JobKind::Heal, 1, 1)));
    }

    #[test]
    fn clear_reports_dropped_and_empties() {
        let q = JobQueue::default();
        q.enqueue(job(JobKind::Heal, 0), false).unwrap();
        q.enqueue(job(JobKind::Heal, 1), false).unwrap();
        assert_eq!(q.clear().unwrap(), 2);
        assert!(q.is_empty().unwrap());
    }

    #[test]
    fn try_start_wins_once_until_cleared() {
        let q = JobQueue::default();
        assert!(q.try_start()); // false -> true: wins
        assert!(!q.try_start()); // already true: loses
        q.clear_running();
        assert!(q.try_start()); // false again: wins
    }

    #[test]
    fn export_kind_is_a_distinct_job() {
        let q = JobQueue::default();
        assert!(q.enqueue(job(JobKind::Heal, 1), false).unwrap());
        assert!(q.enqueue(job(JobKind::Export, 1), false).unwrap()); // distinct kind, kept
        assert!(!q.enqueue(job(JobKind::Export, 1), false).unwrap()); // duplicate coalesced
        assert_eq!(q.len().unwrap(), 2);
    }

    #[test]
    fn prefetch_kind_coalesces_separately_from_other_kinds_on_the_same_index() {
        let q = JobQueue::default();
        assert!(q.enqueue(job(JobKind::Detect, 1), false).unwrap());
        assert!(q.enqueue(job(JobKind::Heal, 1), false).unwrap());
        assert!(q.enqueue(job(JobKind::Export, 1), false).unwrap());
        // Prefetch on the same index as three other kinds is a distinct job.
        assert!(q.enqueue(job(JobKind::Prefetch, 1), false).unwrap());
        // A second prefetch on that index coalesces with the first, not with
        // any of the other kinds.
        assert!(!q.enqueue(job(JobKind::Prefetch, 1), false).unwrap());
        assert_eq!(q.len().unwrap(), 4);
    }

    #[test]
    fn remove_takes_out_exactly_the_matching_job() {
        let q = JobQueue::default();
        q.enqueue(job(JobKind::Heal, 1), false).unwrap();
        q.enqueue(job(JobKind::Export, 1), false).unwrap();
        // Wrong kind, wrong index, wrong generation: all no-ops.
        assert!(!q.remove(job(JobKind::Detect, 1)).unwrap());
        assert!(!q.remove(job(JobKind::Heal, 2)).unwrap());
        assert!(!q.remove(job_gen(JobKind::Heal, 1, 9)).unwrap());
        assert_eq!(q.len().unwrap(), 2);
        // Exact match removes only that job.
        assert!(q.remove(job(JobKind::Heal, 1)).unwrap());
        assert_eq!(q.snapshot().unwrap(), vec![job(JobKind::Export, 1)]);
        // Removing it again is a no-op.
        assert!(!q.remove(job(JobKind::Heal, 1)).unwrap());
    }

    #[test]
    fn drain_returns_the_jobs_it_empties_in_order() {
        let q = JobQueue::default();
        q.enqueue(job(JobKind::Detect, 0), false).unwrap();
        q.enqueue(job(JobKind::Heal, 1), false).unwrap();
        assert_eq!(
            q.drain().unwrap(),
            vec![job(JobKind::Detect, 0), job(JobKind::Heal, 1)]
        );
        assert!(q.is_empty().unwrap());
        assert!(q.drain().unwrap().is_empty());
    }

    #[test]
    fn cancel_targets_only_the_job_that_is_actually_running() {
        let q = JobQueue::default();
        // Nothing running: a cancel request lands nowhere.
        assert!(!q.request_cancel(job(JobKind::Heal, 1)));
        assert!(!q.cancel_requested());

        q.begin_job(job(JobKind::Heal, 1));
        // Mismatches never set the flag.
        assert!(!q.request_cancel(job(JobKind::Heal, 2)));
        assert!(!q.request_cancel(job(JobKind::Export, 1)));
        assert!(!q.request_cancel(job_gen(JobKind::Heal, 1, 9)));
        assert!(!q.cancel_requested());
        // The exact running job sets it.
        assert!(q.request_cancel(job(JobKind::Heal, 1)));
        assert!(q.cancel_requested());
        q.end_job();
        assert!(!q.cancel_requested());
    }

    #[test]
    fn begin_job_clears_a_stale_cancel_from_the_previous_job() {
        let q = JobQueue::default();
        q.begin_job(job(JobKind::Heal, 1));
        assert!(q.request_cancel(job(JobKind::Heal, 1)));
        // The worker moves on without end_job (e.g. the job noticed the flag
        // and returned): the next begin_job must not inherit the request.
        q.begin_job(job(JobKind::Heal, 2));
        assert!(!q.cancel_requested());
    }

    #[test]
    fn request_cancel_running_cancels_whatever_runs() {
        let q = JobQueue::default();
        assert!(!q.request_cancel_running());
        q.begin_job(job(JobKind::Export, 3));
        assert!(q.request_cancel_running());
        assert!(q.cancel_requested());
    }

    #[test]
    fn pin_round_trips() {
        let q = JobQueue::default();
        assert_eq!(q.pinned().unwrap(), None);
        q.pin(Some(42)).unwrap();
        assert_eq!(q.pinned().unwrap(), Some(42));
        q.pin(None).unwrap();
        assert_eq!(q.pinned().unwrap(), None);
    }

    #[test]
    fn snapshot_returns_queue_order_without_draining() {
        let q = JobQueue::default();
        q.enqueue(job(JobKind::Detect, 0), false).unwrap();
        q.enqueue(job(JobKind::Heal, 1), false).unwrap();
        q.enqueue(job(JobKind::Export, 2), false).unwrap();

        let snap = q.snapshot().unwrap();
        assert_eq!(
            snap,
            vec![
                job(JobKind::Detect, 0),
                job(JobKind::Heal, 1),
                job(JobKind::Export, 2),
            ]
        );
        // Snapshot must not drain: pop still returns the same front-first order.
        assert_eq!(q.pop().unwrap(), Some(job(JobKind::Detect, 0)));
        assert_eq!(q.pop().unwrap(), Some(job(JobKind::Heal, 1)));
        assert_eq!(q.pop().unwrap(), Some(job(JobKind::Export, 2)));
        assert_eq!(q.pop().unwrap(), None);
    }
}
