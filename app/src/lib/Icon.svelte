<script module lang="ts">
  export type IconName =
    | "scan"
    | "roll"
    | "detect"
    | "heal"
    | "export"
    | "approve"
    | "unapprove"
    | "download"
    | "paint"
    | "erase"
    | "overlay";

  const paths: Record<IconName, string> = {
    scan: "M4 5h16v14H4z M4 9h16",
    roll: "M3 7h13v10H3z M16 10h5v4h-5z M6 7v10 M13 7v10",
    detect: "M11 4a7 7 0 1 0 0 14 a7 7 0 0 0 0-14 M16 16l4 4",
    heal: "M12 4v16 M4 12h16",
    // Up-and-out over a base line -- the mirror of download's down-into-a-
    // tray, so the two no longer differ by a single pixel of baseline y.
    export: "M12 14V4 M8 8l4-4 4 4 M5 18h14",
    approve: "M5 13l4 4 10-10",
    unapprove: "M6 6l12 12 M18 6L6 18",
    download: "M12 4v10 M8 10l4 4 4-4 M5 20h14",
    overlay: "M12 6c-4 0-7 3-8 6 1 3 4 6 8 6s7-3 8-6c-1-3-4-6-8-6 M12 10a2 2 0 1 0 0 4 a2 2 0 0 0 0-4",
    paint: "M14 4l6 6-9 9H5v-6z M12 6l6 6",
    erase: "M9 19h11 M6 14l7-7 5 5-7 7H8z",
  };
</script>

<script lang="ts">
  let { name }: { name: IconName } = $props();
</script>

<svg
  viewBox="0 0 24 24"
  width="14"
  height="14"
  fill="none"
  stroke="currentColor"
  stroke-width="1.5"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
>
  <path d={paths[name]} />
</svg>
