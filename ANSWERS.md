# Day 1 Lab: Answers

Name: Ghulam Ali

## 1. Leakage
I dropped the columns in `LEAKY`: `title`, `sig`, `mmi`, `cdi`, `felt`, `alert`, `tsunami`, and `magType`. These fields either directly reveal the target outcome or are post-event metadata that would not be known at prediction time. `tsunami` is leaky because it is an outcome signal tied to the quake's impact, not a causal input. `magType` is not a direct target leak, but it is measurement metadata that can encode how the magnitude was reported and can reduce generalization, so it was dropped as part of the leakage guardrail.

## 2. Stream vs batch
In the local run, the stream merge count was `0` updated events: no duplicate `id` values with a newer `updated` timestamp appeared in the available stream data. That supports the principle of “latest version wins”: when updates do occur, the newest record should replace the older one, and the deduplication logic keeps the freshest version without duplicate IDs.

## 3. Outliers
I kept negative depth and negative magnitude unless they are obviously impossible or extreme beyond the data-quality rules. Real seismic feeds can contain small negative values due to measurement noise, reporting quirks, or special cases, so deleting them blindly would throw away valid data. The safer approach is to use an IQR-based flag for hard outliers and only remove the truly extreme values, rather than dropping all negative numbers.

## 4. Cardinality
One alternative to grouping `region` to top-k is one-hot encoding the full set of regions with a learned or frequency-based grouping, or using Target Encoding. A risk is that rare regions can create many sparse columns or unstable estimates, which may overfit the training data and hurt generalization if the rare categories are not well represented.

