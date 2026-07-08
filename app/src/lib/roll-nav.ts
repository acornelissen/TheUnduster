/** Next unapproved frame index after `from`, wrapping around the roll so
 * approve-and-advance always lands on remaining work regardless of the
 * order the operator approved in. -1 when everything is approved. Never
 * returns `from` itself. */
export function nextUnapprovedIndex(approved: boolean[], from: number): number {
  const n = approved.length;
  for (let step = 1; step < n; step++) {
    const i = (from + step) % n;
    if (!approved[i]) return i;
  }
  return -1;
}
