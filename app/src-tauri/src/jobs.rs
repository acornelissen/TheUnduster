//! Per-roll background job queue state.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Detect,
    Heal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Job {
    pub kind: JobKind,
    pub index: usize,
}

/// Per-roll job queue. One worker drains it (single-flight `running` flag,
/// drop-guard cleared); `pinned` is the image id of the frame the active
/// job is processing, which eviction must skip.
#[derive(Default)]
pub struct JobQueue {
    queue: Mutex<VecDeque<Job>>,
    pub running: AtomicBool,
    pinned: Mutex<Option<u64>>,
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

    /// Returns the number of jobs currently in the queue.
    // Exercised by tests; no production caller yet.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn len(&self) -> Result<usize, String> {
        let queue = self.queue.lock().map_err(|e| e.to_string())?;
        Ok(queue.len())
    }

    /// Returns true if the queue is empty.
    // Exercised by tests; no production caller yet.
    #[cfg_attr(not(test), allow(dead_code))]
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(kind: JobKind, index: usize) -> Job {
        Job { kind, index }
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
    fn clear_reports_dropped_and_empties() {
        let q = JobQueue::default();
        q.enqueue(job(JobKind::Heal, 0), false).unwrap();
        q.enqueue(job(JobKind::Heal, 1), false).unwrap();
        assert_eq!(q.clear().unwrap(), 2);
        assert!(q.is_empty().unwrap());
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
}
