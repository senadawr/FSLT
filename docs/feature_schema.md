# Feature schema: `fsl-landmarks-v1`

Each frame contains 553 landmarks in fixed order: 478 face, 33 pose, 21 left
hand, and 21 right hand. `features` contains 3 values per landmark in
`[x, y, z]` order, for 1,659 values. Coordinates are MediaPipe normalized
coordinates and are not mirrored by the UI preview. `presence` contains one
value per landmark: `1` when detected and `0` when missing.

Missing coordinates are zero-filled, but presence is mandatory so zero is never
interpreted as a stationary detected landmark. The schema is versioned and the
backend rejects unknown versions.
