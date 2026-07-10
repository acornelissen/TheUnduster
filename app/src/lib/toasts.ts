/** Pure list helpers for the toast stack and activity log. App.svelte owns
 * the reactive arrays (`$state`) and calls these on every mutation; kept
 * here so the collapsing/capping logic has direct test coverage without
 * mounting Svelte. */

export interface Toast {
  id: number;
  level: "info" | "error";
  message: string;
  count: number; // collapsed repeats
}

/** Append a toast, or collapse into the last toast (incrementing its count,
 * keeping its id) when both level and message match the last entry. Only
 * the last toast is checked -- an earlier matching toast further back in
 * the stack is left alone and a new entry is appended instead. */
export function pushToast(
  list: Toast[],
  level: Toast["level"],
  message: string,
  id: number,
): Toast[] {
  const last = list[list.length - 1];
  if (last && last.level === level && last.message === message) {
    return [...list.slice(0, -1), { ...last, count: last.count + 1 }];
  }
  return [...list, { id, level, message, count: 1 }];
}

/** Remove the toast with the matching id. */
export function dismissToast(list: Toast[], id: number): Toast[] {
  return list.filter((t) => t.id !== id);
}

/** Append `entry`, dropping the oldest entries so the result never exceeds
 * `cap`. */
export function pushLog<T>(list: T[], entry: T, cap: number): T[] {
  const next = [...list, entry];
  return next.length > cap ? next.slice(next.length - cap) : next;
}
