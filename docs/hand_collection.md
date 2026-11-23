# Hand landmark collection

The browser uses MediaPipe Tasks Vision Hand Landmarker to track 21 landmarks
for each hand. Camera frames and saved poses remain local to the project. The
live translator stores one normalized pose per user-defined label in
`artifacts/pose_templates.json` through the local backend and reports the
nearest-label similarity as confidence. Existing browser-only templates are
migrated automatically on startup.

## Save a pose

1. Start the camera.
2. Hold the sign pose steadily.
3. Enter its label and click **Save current pose**.
4. Repeat for each label. Saving an existing label replaces its pose.

Left-hand input is mirrored into the right-hand coordinate convention before
matching, so right-hand templates can be used with the left hand.

The matcher is intended for static, one-hand poses. It does not learn motion,
timing, or continuous sign-language sentences.
