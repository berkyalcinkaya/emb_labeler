# Improve Time Point Labeling

This document outlines several UI features to streamline the Time Point Labeling process.

## Time Point Chart — anomaly & confidence filters

1. **Argmax vs. decoding mismatch** — Do not surface by default. Add a checkbox to show these anomalies only; toggled **off** by default.
2. **Low confidence** — Also toggled **off** by default.

These checkboxes should be small at the bottom left corner of the UI beneath timepoint chart. 

## Navigation

3. **Scroll to predicted time point boundaries** — `Cmd+Right` / `Cmd+Left` jumps to the next / previous boundary as per predicted timepoint.

4. **Selecting a timepoint auto-advances timepoint** - User labeling a time point, either by buttons on the right pane or by keyboard shortcut, moves UI to the next time point.

## Auto-fill

5. **Remove Bracket auto-fill** — Remove the existing `[` / `]` bracket behavior ; too clunky
6. **"Accept predicted"** — When invoked, On time points where the user has not set a manual label, keep auto-filling with the predicted time point label. Expose  **keyboard shortcut** with the key F 
7. **Auto-fill between labels** - When invoked, fill from the current time point to the previous labeled time point with the selected label, assuming the two labels are consistent. This would allow a user to fill gaps between two labeled time points that each have the same label. This should be a no-op if 
- the labels differ between the current label and the previous labeled time point.
- the current timepoint has no label
Edge case: if there is no previously labeled timepoint, autofill to t=0 with current label
Keyboard shortcut: Cmd+F
