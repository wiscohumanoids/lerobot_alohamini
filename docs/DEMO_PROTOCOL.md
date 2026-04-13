# Demonstrator Protocol

This document describes how to record clean, consistent demonstrations for training ACT (Action Chunking Transformer) policies on the AlohaMini. It is intended for anyone who records teleoperation data, including new team members who have never touched the robot before. Read it once end-to-end before your first recording session, and keep the per-task template handy when you sit down at the leader arms.

## Why Demo Consistency Matters

ACT learns to mimic the demonstrator. It does not learn the abstract idea of "pick up the cube" - it learns the specific trajectories, speeds, and grasps that appear in the training data. If two operators record "the same task" with subtly different strategies, the resulting dataset contains a multimodal action distribution: at any given state, there are multiple plausible next actions, and the policy ends up learning something close to the average of them. Averaging two different grasp approaches usually produces a third approach that grasps nothing.

The practical consequence is that **50 clean, consistent demos almost always beat 200 noisy, inconsistent ones**. A dataset where every episode starts in the same pose, uses the same grasp strategy, moves at the same speed, and ends the same way is dramatically easier to train on than a larger dataset with inconsistent behavior. Before you record your first episode of a new task, define the protocol. Before you record your second episode, make sure it matches the first. This document exists so that the protocol is written down and every demonstrator follows it.

Consistency is most important in the parts of the task where the policy has the least room for error: contact events, grasp transitions, and the final approach to a target. It matters less in free-space travel between waypoints, where small variations actually help the policy generalize. The right mental model is: **be boring where precision matters, and stay natural everywhere else**. If you find yourself freezing up trying to perfectly replicate a trajectory, you are overcorrecting. The goal is a dataset that looks like the same person doing the same task many times, not a dataset that looks like a robot doing a pre-recorded animation.

## What to Hold Constant and What to Vary

Consistency does not mean zero variation. The goal is a dataset where unhelpful variation is removed and helpful variation (the kind you want the policy to generalize over) is preserved. Decide these before you record, not during.

**Hold constant:**

- Start pose of both arms.
- Base position and heading (unless base motion is part of the task).
- Camera mount positions.
- Grasp strategy for each object.
- Overall task structure and ordering of sub-steps.
- Demonstrator style: who is holding the leader arms, and how.

**Vary deliberately:**

- Object position within the marked placement zone (not outside it).
- Object orientation, if the task should be orientation-robust.
- Small, natural variations in approach speed and trajectory shape. Do not try to draw the exact same arc every time; that produces overfit, jerky policies.
- Ambient noise in the scene (people walking by in the background), but only if you want the policy to tolerate it.

**Do not vary casually:**

- Lighting.
- Backgrounds and distractors.
- The presence or absence of additional objects on the workspace.
- Which demonstrator is recording, within a single dataset. If you need multiple demonstrators, plan that deliberately and balance episodes across them.

The principle is: vary the things you want the policy to handle at deployment, and lock down everything else. Unintended variation is the enemy.

## Pre-Recording Checklist

Run through this list at the start of every recording session, before you touch a leader arm:

- Robot is calibrated. Confirm that `AlohaMiniRobot.json` exists in `calibration/robot/`. If it does not, run calibration before recording any data.
- Cameras are enabled in `config_lekiwi.py`. Only the cameras you actually intend to use should be active - extra cameras add noise to the dataset and slow things down.
- All enabled cameras are visible in the Rerun preview and the image feeds look correct (no black frames, no frozen images, no wrong orientation).
- Camera mounts are physically stable. A loose camera that shifts mid-session invalidates every subsequent episode.
- Leader arm USB ports are configured correctly in `config_lekiwi.py`. Left and right should not be swapped.
- Battery is charged, or the motor boards are plugged into the wall outlet. If you are running on battery, note the starting voltage so you can correlate any late-session motor dropouts with battery sag.
- Workspace lighting matches the lighting you expect at deployment time. Do not record under bright overhead lights if the robot will eventually run in a dimmer room. If the room has windows, record with the blinds in a consistent position.
- Object and target locations are physically marked on the table (colored tape, printed marker, or similar). A "roughly here" placement is not good enough.
- The Jetson `lekiwi_host.py` process is running and the client on your laptop connects cleanly with no stale-ZMQ warnings. See `RECORD.md` for startup steps.
- Rerun is launched (`rerun --web-viewer-port 9091`) and you can see the live video feeds before the first episode.
- You have a clear plan for how many episodes you intend to record and roughly how long the session will take. Going in without a target invites drift.

If any of these are not true, fix them before recording. Do not "just get a few demos in" with a broken setup - the data will be unusable.

## Per-Task Protocol Template

Copy the template below for each new task you record. Fill in every field before the first episode. Commit the filled-in protocol alongside the dataset so future demonstrators (including you, three months from now) know exactly what you did.

```
Task name: <short identifier, e.g. "pick_place_cube_right_to_left">
Description: <one or two sentences of what the robot is doing>

Success criteria:
  - <explicit, observable condition 1>
  - <explicit, observable condition 2>
  - <when is the episode considered "done"?>

Start configuration:
  - Arm pose: <home / custom joint angles / photo reference>
  - Base pose: <stationary / facing target / specific heading>
  - Gripper state: <open / closed>

Object placement zone:
  - Object: <what it is>
  - Location: <tape mark, coordinates, photo reference>
  - Tolerance: <allowed offset, e.g. +/- 1 cm>
  - Orientation: <any / fixed / within X degrees of fixed>

Allowed grasp strategies:
  - <top grasp only / side grasp allowed / either / bimanual>
  - <if multiple are allowed, which episodes use which?>

Target speed:
  - <deliberate (~3-5s) / natural (~1-2s) / slow and careful (~5-10s)>
  - Consistency is more important than the exact number.

Mid-episode failure handling:
  - <abort and re-record / attempt in-episode recovery>
  - If recovery is allowed, describe exactly what counts as recovery.

Reset checklist (between episodes):
  - <step 1>
  - <step 2>
  - <wait time before starting the next episode>
```

Decide each field once, at the start of a recording batch, and stick with it. If you genuinely need to change the protocol (for example, you realized the start pose is unreachable), throw away the episodes recorded under the old protocol or put them in a separate dataset. Mixing protocols inside one dataset is the fastest way to ruin a training run.

## Thinking About Task Scope

Before you write a per-task protocol, decide whether the thing you want the robot to do is actually one task or several. A good rule: if the set of "allowed strategies" has more than one or two entries, it is probably two tasks glued together.

Consider "clean the table." That sentence describes at least four different behaviors: wiping with a cloth, picking up small objects, stacking dishes, and pushing debris off the edge. An ACT policy trained on a single dataset that mixes all four will learn the confused average of all of them and do none well. The fix is to split into four protocols and four datasets. You can still use them together at deployment time (via a higher-level task selector) but each dataset on its own needs to represent one coherent behavior.

Some signs that your task is too broad:

- You find yourself writing "or" in the allowed grasp strategies section more than once.
- Success criteria depend on which variant of the task the demonstrator chose.
- Episode durations span more than a 3x range across the dataset.
- Two demonstrators watching the same protocol produce visibly different trajectories and both feel justified.

When in doubt, split. It is easier to merge two narrow datasets later (by training on their union) than to untangle one contaminated dataset.

## Structuring a Recording Session

A recording session is more than just "sit down and record 100 episodes." Plan it in phases:

1. **Setup and warmup (5-10 minutes).** Run the pre-recording checklist. Then record 2-3 throwaway episodes to get your hands used to the leader arms and verify that the full pipeline is working end-to-end. Delete these episodes afterwards - do not keep them.
2. **Protocol review (2-3 minutes).** Re-read the per-task protocol, even if you wrote it yourself. Look at the success criteria. Place the object at the starting location and visualize the trajectory you are about to execute.
3. **Recording block (20-30 minutes, 20-30 episodes).** Record in focused blocks. After each block, stand up, stretch, and reset your attention. Fatigue is the single biggest source of inconsistency within a session.
4. **Spot-check (2-3 minutes).** Every 2-3 blocks, play back a few recent episodes in Rerun or the dataset viewer. Look for drift - are episodes slowly getting faster, sloppier, or shorter than the ones you recorded at the start? If so, stop, reset, and re-read the protocol.
5. **Session end.** Note anything unusual about the session in the dataset's README or commit message: lighting changes, battery swaps, new demonstrators, anything that distinguishes this batch from previous ones.

Do not try to record 200 episodes in a single sitting. Human attention fades quickly when the task is repetitive, and the last 50 episodes of a marathon session are almost always worse than the first 50. Two 100-episode sessions on different days produce better data than one 200-episode session.

## General Teleop Best Practices

These apply to every task, regardless of the specific protocol:

- Move slowly and deliberately, especially during contact-rich phases like grasping, inserting, or placing. ACT handles slow motion fine and struggles with high-speed contact.
- Avoid fast direction reversals. Sudden jerks produce high-frequency noise in the action signal that the policy will either learn (badly) or smooth over (also badly).
- Do not hold the leader arm at its joint limits. If you are pushing into a mechanical stop, the follower will see one value while the leader continues to register motion, and the resulting action data is garbage.
- Watch the Rerun preview while recording. Camera dropouts, frozen frames, and incorrect exposure are easy to miss if you are focused on the robot itself.
- Pause briefly at the start and end of each episode. A half-second of stillness gives the policy a clear "begin" and "end" signal and improves the cleanliness of the first and last action chunks.
- Do not talk to the robot mid-episode unless voice control is intentionally part of the task. Microphones can pick up incidental audio and, if audio is ever added to the observation, it will contaminate the dataset.
- Keep your hands out of the cameras' field of view unless hand-in-frame is intentional for the task.
- Keep other people out of the camera frame too. An occasional accidental elbow in the background will make the policy learn to wait for elbows.
- If something feels off mid-episode - a stutter, a lag, a weird grasp - abort and re-record rather than trying to save a marginal episode.
- Do not record while distracted. If you are on a call, in a meeting, or half-watching a stream on a second monitor, stop recording. The dataset will notice.
- When a new demonstrator joins a session, do a side-by-side comparison of their episodes against the existing ones before committing any of their data. Two different hands on the leader arms should produce visually indistinguishable follower behavior.

## Failure Modes To Watch For

The following table lists common problems that show up during recording, their likely causes, and what to do about them.

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Arm jitters or twitches during recording | Stale ZMQ observation data; WiFi or USB dropouts | Check WiFi signal to the Jetson, verify USB cables are seated, restart `lekiwi_host.py` if needed |
| Follower lags noticeably behind the leader | High CPU load on the Jetson or overloaded camera bandwidth | Reduce the number of active cameras, lower resolution, or cap UVC bandwidth with `sudo modprobe uvcvideo bandwidth_cap=800` |
| Gripper slips on the object | Insufficient gripper torque, wrong approach angle, or object too smooth | Re-record with a different grasp approach; do not fight it with "extra squeeze" demos, that only confuses the policy |
| Episode duration varies wildly between runs | Inconsistent task interpretation by the demonstrator | Stop recording, re-read the per-task protocol, agree on a target duration, then restart |
| Episodes look fine individually but the policy still fails | Start pose drifts over time; object placement tolerance too loose | Re-mark start pose and object location with tape, then record a fresh batch |
| Camera feed goes black or freezes in Rerun | USB enumeration changed after a reboot | Use the `/dev/v4l/by-path/` symlinks in `config_lekiwi.py` instead of raw `/dev/videoX` device nodes |
| Motors drop out with "Missing motor IDs" error | Motor board lost power, or battery sagged under load | Power-cycle the battery as described in `RECORD.md`; do not try to record through a flaky power state |
| First few episodes look different from the rest | Demonstrator warming up | Discard the warmup episodes; do not include them in the dataset |
| Leader arm feels sticky or resistant in one joint | Motor friction, loose cable, or servo near thermal limit | Stop, let the arm cool for a few minutes, and check the cable routing before resuming |
| Episode lengths slowly creep upward across a session | Demonstrator fatigue or drifting attention | End the session, rest, and start a fresh batch later |
| Follower gripper closes but the action value never reaches "fully closed" | Calibration range mismatch | Re-run calibration; do not try to compensate by over-squeezing the leader gripper |
| Background changes between episodes (chairs moved, clutter added) | Insufficient workspace control | Physically reset the background before each episode, or at least each batch |

## Reference: Example Filled-In Protocol

The following is a concrete, filled-in protocol for a simple pick-and-place task. Use it as a model for your own tasks.

```
Task name: pick_place_cube_right_to_left
Description: Right arm picks up a 4 cm foam cube from the green tape mark and places it on the red tape mark. Left arm is idle throughout.

Success criteria:
  - Cube is fully released and resting on the red tape mark.
  - Cube is not touching the right gripper at the end of the episode.
  - Both arms are returned to the home pose before the episode ends.
  - Episode ends with a half-second pause in the home pose.

Start configuration:
  - Arm pose: both arms at home pose (see calibration/robot/AlohaMiniRobot.json)
  - Base pose: stationary, centered on the workspace
  - Gripper state: both grippers open

Object placement zone:
  - Object: 4 cm blue foam cube
  - Location: centered on the green tape mark on the right half of the table
  - Tolerance: +/- 1 cm in X and Y
  - Orientation: any (cube is symmetric)

Allowed grasp strategies:
  - Right arm only, top grasp, approach straight down from above
  - Fingers aligned with the cube's long edge
  - No side grasps, no pinch-from-corner grasps

Target speed:
  - Deliberate (~3-5 seconds per episode total)
  - Approach: ~1s, grasp: ~0.5s, lift and translate: ~1.5s, place and release: ~1s, return home: ~1s

Mid-episode failure handling:
  - Drop the cube: abort and re-record. Do not pick it up again mid-episode.
  - Miss the red mark by more than 2 cm: abort and re-record.
  - Collision with the table: abort and re-record.

Reset checklist (between episodes):
  - Place cube back on the green tape mark, aligned to within 1 cm.
  - Return both arms to home pose using the leader arms.
  - Confirm gripper state is open.
  - Confirm Rerun preview shows live video from all enabled cameras.
  - Wait 2 seconds in the home pose before starting the next episode.
```

## After the Session: Data Hygiene

Recording is only half the job. Before the dataset is considered ready for training:

- Review every episode at least briefly in the dataset viewer. You are looking for obvious glitches: frozen frames, dropped motor data, episodes that end in the wrong pose, episodes where the object was not in the expected start location.
- Delete bad episodes aggressively. If you are on the fence about whether an episode is clean, delete it. A smaller clean dataset is always better than a larger contaminated one.
- Tag or note any unusual sessions in the dataset metadata (see the `tasks` and `meta` files in the LeRobot dataset format). Future you will want to know that episodes 200-250 were recorded by a different demonstrator under different lighting.
- If the dataset spans multiple recording sessions, spot-check the transition: do episodes from session 2 look like episodes from session 1? If not, investigate before training.
- Commit the filled-in per-task protocol alongside the dataset, or link it from the dataset README. The protocol and the data belong together.

## How Many Episodes Do I Need?

There is no universal answer, but here are some rules of thumb from our own training runs:

- **Simple single-arm pick and place** (cube to target, no distractors): start with 50 episodes. If the policy trains cleanly but generalizes poorly, add another 50 with wider object placement variation.
- **Bimanual tasks** (handovers, two-arm manipulation): start with 100 episodes. Bimanual coordination is harder for the policy to learn and needs more data.
- **Tasks with visual variation** (different colored objects, different lighting conditions): add ~20 episodes per variation axis. Do not spread variation uniformly - pick the specific axes you care about and cover them deliberately.
- **Tasks involving the mobile base**: add episodes that cover the full range of base positions and headings you expect at deployment. Static-base tasks are much easier than mobile tasks.

If the policy is failing and your first instinct is "we need more data," pause and ask the harder question first: is the existing data consistent? Ten fresh, consistent episodes can sometimes rescue a failing policy faster than a hundred more inconsistent ones. Data quality compounds. Data quantity plateaus.

## Closing Notes

When a new demonstrator joins the team, point them at this document, walk them through the example protocol, and record a handful of episodes with them watching before you let them run solo. The hour you spend on onboarding will save you days of debugging a policy that learned from inconsistent data.

If you find yourself diverging from this protocol repeatedly for a specific task, that is a signal that the protocol needs to be updated, not ignored. Open a PR to this document, describe what you changed and why, and get agreement from the rest of the team before the next recording session. Written protocols are only useful if they reflect what demonstrators actually do.
