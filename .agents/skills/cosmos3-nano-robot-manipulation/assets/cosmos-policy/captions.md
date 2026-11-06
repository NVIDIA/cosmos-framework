# Cosmos Policy recipe media

All media are frames and clips from the closed-loop LIBERO-10 evaluations reported in the [recipe](../../README.md);
nothing was regenerated. Frames show the third-person camera (the left half of the stored 512×256 third-person | wrist
video), 256×256 upscaled to 512. Labels are burned in: task, iteration, seed, outcome. "Steps" are control steps at
20 per second; failures time out at 520.

The iteration-2000 media come from evaluation seed 0 of the iteration-2000 checkpoint, evaluated on a second GB300. The
three-seed mean in the recipe (95.1 ± 0.1 %) was measured on the GB300 that trained it.

## Hero clip

| File                                                                             | Content                                                                                                                                                                                                                                                                                          |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `hero_iter2000_task5_success_vs_failure.gif` (about 1.4 MB, 10 fps, 640 px wide) | Side by side: task 5, "pick up the book and place it in the back compartment of the caddy", iteration 2000, evaluation seed 0. Left: success from initial state 0 (176 steps). Right: failure from initial state 3 (timed out at 520 steps). First 15 s; the success panel holds its last frame. |
| `hero_iter2000_task5_success_vs_failure.mp4` (20 fps, 768×436, H.264)            | The same clip.                                                                                                                                                                                                                                                                                   |

## Result frames (last frame of the episode)

| File                                         | Task                                                                 | Iteration                                                                                                                                                                                                                                                                    | Seed | Outcome                       |
| -------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ----------------------------- |
| `frame_iter2000_task1_success.png`           | 1, put both the cream cheese box and the butter in the basket        | 2000                                                                                                                                                                                                                                                                         | 0    | success, 259 steps            |
| `frame_iter2000_task7_success.png`           | 7, put both the alphabet soup and the cream cheese box in the basket | 2000                                                                                                                                                                                                                                                                         | 0    | success, 260 steps            |
| `frame_iter2000_task8_failure.png`           | 8, put both moka pots on the stove                                   | 2000                                                                                                                                                                                                                                                                         | 0    | failure, 520 steps (time-out) |
| `frame_perturbed_iter1250_task0_success.png` | 0, put both the alphabet soup and the tomato sauce in the basket     | 1250, from a separate restyle study, not part of this recipe: the iteration-1000 checkpoint trained 250 more iterations on the 379 demonstrations plus 187 restyled ones (training seed 42). **Perturbed scene**: randomised lighting, colours and textures, jittered camera | 0    | success, 312 steps            |

Task names come from each episode's metadata; the file names use the LIBERO-10 task index. `_manifest.json` lists
the task, iteration, seed, initial state and outcome of every file.

## Not available

- **Iteration 0 (base model) and iteration 500:** no evaluation videos were saved, so these checkpoints have numbers
  in the recipe but no frames.
