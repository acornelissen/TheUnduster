# TheUnduster user manual

TheUnduster finds dust, scratches, and hairs on scanned film and removes them. Open a single scan or a whole roll, review what it finds, adjust by hand where needed, then heal and export.

## Getting started

The first time you open TheUnduster, the healing model isn't on your computer yet. You'll see a "Download healing model (207 MB)" button in the toolbar. Click it when you're ready — the button shows progress as it downloads. You can keep working while it downloads; you just won't be able to heal anything until it finishes. If the download fails partway, the button reappears so you can try again.

Until the model is downloaded, healing still works, but it uses a simpler fill instead of the full neural repair, so results are lower quality on anything but small specks.

## Opening your scans

You can open a single scan or a whole roll (a folder of scans).

- **Single scan**: click "Open scan" in the toolbar, or use the File menu ("Open Scan...", Cmd+O).
- **Roll**: click "Open roll" in the toolbar, or use the File menu ("Open Roll...", Cmd+Shift+O). Pick the folder that contains your scanned frames.
- **Drag and drop**: drag a single image file or a single folder anywhere onto the window. Dropping more than one item at once isn't supported — drop one scan or one folder.

When nothing is open, the window shows an empty state with a hint to drop a scan or folder in.

## Finding the dust

Click "Detect" (or press `d`) to scan the current frame for defects. Once detection has run, red circles appear over anything it found, and a light red tint washes over the whole image.

The sensitivity slider controls how much gets flagged. Drag it and the circles and the defect count update live, without needing to press Detect again. While the app is recalculating after you move the slider, the circles dim briefly — that's normal, it means new results are on the way, not that anything is wrong.

After a frame is detected, the Detect button greys out to "Detected" — there's nothing to re-run; the slider works on the saved results.

Press `m` to toggle the red tint and circles on or off without losing your detection results. Press `z` or `shift-z` to jump the view from one flagged defect to the next; the circle you're on turns amber so you can tell which one is selected. Press `delete` or `backspace` to remove it — this paints an erase stroke over the spot (undoable like any other stroke) and drops the circle from view.

## Fixing by hand

If the automatic detection misses something or catches something it shouldn't, you can paint or erase the area yourself.

- Press `b` to start painting a mask over a defect, or `e` to erase part of an existing mask.
- Use the `[` and `]` keys to shrink or grow your brush size.
- A white ring around your cursor shows exactly where the brush will paint.
- Press `Escape` to stop painting or erasing.
- Cmd+Z undoes your last stroke; Shift+Cmd+Z redoes it.

The Paint, Erase, and Overlay buttons in the bottom-left corner of the image do the same as the `b`, `e`, and `m` keys, and show which mode is active. While a brush is on, your current brush size appears next to them.

Your brush strokes are saved automatically as part of the frame's working state.

## Healing

Click "Heal" (or press `h`) to repair the current frame — both the automatically detected defects and anything you painted by hand. Healing can take a little while, especially on a large scan, and shows its progress in the status bar.

Once a frame is healed, press `space` to flip between the original and the healed version, so you can check the repair before moving on.

## Working through a roll

When you open a roll, a filmstrip of thumbnails appears along the bottom. Use `,` and `.` to move to the previous or next frame, or click a thumbnail directly.

Press `a` to approve the current frame — marking it done and ready to export. Move on with `,` and `.` when you're ready.

Changed your mind? Press `shift-a`, or click the "Unapprove" button that replaces "Approve" on an approved frame, to un-approve it. This doesn't advance to another frame — you stay put.

Each thumbnail can show small badges in its corners:

- A checkmark means the frame is approved.
- "out" means the frame has already been exported.
- A colored dot in the bottom-left corner means work is queued or running on that frame: grey for detection (and for the automatic background prefetching that warms up nearby frames as you browse), amber for healing, blue for exporting. A hollow dot means the work is waiting in the queue; a solid, pulsing dot means it's running right now.

The status bar shows a count of how many frames are approved out of the total, and how many jobs are currently queued.

## The queue and the log

TheUnduster processes detection, healing, and export jobs in the background so you can keep working. Two buttons in the status bar, "Queue" and "Log", open side panels:

- **Queue** shows every job waiting to run or currently running, with a progress bar for whichever job is active.
- **Log** shows a running history of what's happened — completions, errors, and other notable events, newest first.

Only one of these panels (or the keyboard shortcuts panel) is open at a time; opening one closes the others.

## Exporting

For a single scan, click "Export" once it's healed. You'll be asked where to save it; the file is saved in the same format you opened.

For a roll, click "Export approved" to export every frame you've approved so far, to a folder you choose. If a frame you approved hasn't been healed yet, TheUnduster heals it automatically as part of the export.

If you approve more frames later, click "Export approved" again — it re-exports everything currently approved, including frames you exported before, so anything that changed is brought up to date.

## Keyboard shortcuts

Press `?` at any time to open the in-app shortcuts panel. It's the authoritative list; here it is in full.

**Viewer**

| Key | Action |
| --- | --- |
| `d` | Detect |
| `h` | Heal |
| `space` | Before/after (healed) |
| `m` | Toggle overlay |
| `z` / `shift-z` | Cycle through defects |
| `delete` / `backspace` | Delete the selected defect (paints an erase stroke) |
| `+` / `-` | Zoom in/out |
| `0` | Fit to window |
| `1` | 100% zoom |
| arrows | Pan |

**Brush**

| Key | Action |
| --- | --- |
| `b` | Paint |
| `e` | Erase |
| `[` / `]` | Brush size |
| arrows | Nudge brush (hold shift for faster) |
| `enter` | Stamp a single dot |
| `esc` | Exit brush mode |

**Roll**

| Key | Action |
| --- | --- |
| `,` / `.` | Previous/next frame |
| `a` | Approve the current frame |
| `shift-a` | Unapprove |

**Everywhere**

| Key | Action |
| --- | --- |
| `cmd-z` | Undo |
| `shift-cmd-z` | Redo |
| `?` | Open this shortcuts panel |
| `esc` | Close open panels |

## Tips

- When no scan is open, the empty state reminds you that you can drop a scan or a roll folder anywhere in the window.
- Use the `+`/`-` keys, the zoom buttons in the bottom-right corner of the image, or your trackpad's pinch gesture to zoom. `0` fits the whole image to the window; `1` jumps to 100%.
- If you have Reduce Motion turned on in macOS, TheUnduster respects it — the filmstrip skips its smooth-scrolling animation and jumps straight to the selected frame instead.

## Current limits

- Detection quality depends on the trained model behind it. Right now the app ships with a placeholder detector for development, so what gets flagged is not yet representative of the finished product. This will improve as a properly trained model ships.
- Healing needs the downloaded model to do its best work. Without it, TheUnduster falls back to a simpler fill that works fine for tiny specks but won't hold up on larger scratches or hairs.
