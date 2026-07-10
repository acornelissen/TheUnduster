export type PathKind = "file" | "dir";

export type DropRoute = { action: "scan" | "roll"; path: string } | { error: string };

/** Routes a set of dropped paths to a scan open or a roll open. Pure: the
 * caller stats each path (via the `path_kind` backend command) and passes
 * the resulting kinds in, so this stays a plain function with no invoke
 * inside and no async boundary to test around. Only a single dropped path
 * is ever wired to an open; anything else is a routing error, not a partial
 * multi-open. */
export function routeDrop(paths: string[], kinds: PathKind[]): DropRoute {
  if (paths.length !== 1) {
    return { error: "drop a single scan or one roll folder" };
  }
  const [path] = paths;
  const [kind] = kinds;
  return { action: kind === "dir" ? "roll" : "scan", path };
}
