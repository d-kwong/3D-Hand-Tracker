# 3D Hand Tracker

This started as a little experiment I made in a day. I wanted to see if I could use my webcam to draw in 3D with my index finger.

MediaPipe's open-source hand model gave me 21 landmarks and an estimate of each landmark's depth relative to my wrist. I still needed to figure out where my fingertip was inside a box I imagined in front of the camera. So I pointed at imaginary locations about 100 times, saved the landmarks each time, and trained a small PyTorch model to predict the position I was aiming for.

**Webcam → MediaPipe landmarks → lightweight PyTorch model → 3D position visualized in OpenGL**

<img src="docs/media/draw-and-rotate.gif" alt="Mirrored hand landmarks alongside a synchronized 3D view, where the estimated fingertip places amber vertices before the finished sketch rotates" width="960">

*On the left is my hand with the MediaPipe landmarks. On the right, the teal dot follows the model’s fingertip estimate. I place vertices and edges, then rotate the finished sketch to see its depth.*

## Collecting training data

<img src="docs/media/collect-samples.gif" alt="Mirrored hand landmarks alongside a synchronized 3D view of the teal target changing position after each training sample" width="960">

*Here the teal dot is the target I’m trying to hit. I press **Space** to pair the current landmarks with that target, then the collector shows a new one. Both panels in each GIF use the same frames from the screen recording.*

## Run it

You'll need Conda, a webcam, and a desktop with OpenGL support. I ran the project on an Apple Silicon Mac. The first camera run downloads MediaPipe's pretrained Hand Landmarker.

```sh
git clone https://github.com/d-kwong/3D-Hand-Tracker.git
cd 3D-Hand-Tracker
conda env create -f environment.yml
conda activate hand-tracker

python collect_data.py   # Collect samples with Space, then exit with Esc
python train.py          # Train on the newest capture
python predict_live.py   # Watch the fingertip prediction
# Or open the same live view with drawing controls
python draw_live.py
```

Captures are timestamped `.jsonl` files in `data/`. Training saves a `.pt` checkpoint alongside the capture. Both live scripts load the most recently saved checkpoint by default. Pass a path to choose a specific capture or checkpoint.

```sh
python train.py data/capture_...jsonl
python draw_live.py data/capture_...pt
```

| Control | Action |
| --- | --- |
| **Space** | Capture a sample in collection mode |
| **1** | Place a vertex at the predicted fingertip |
| **2** | Place a vertex and connect it to the previous one |
| **C** | Clear the sketch |
| **Left-drag** | Pan in the drawing window |
| **Right-drag** | Rotate around the center of the box |
| **Esc** | Close the session |

## How it works

| Stage | Tools | What I built |
| --- | --- | --- |
| Webcam input | OpenCV, MediaPipe Hand Landmarker | Mirror the webcam frame and detect 21 landmarks. Each landmark has normalized image `x` and `y`, plus a wrist-relative `z`. |
| Data collection | Python, JSON Lines | Sample a target inside the box. A keypress pairs the displayed frame’s landmarks with the target’s `[x, y, z]`. Each capture session has its own file. |
| Position model | PyTorch | Flatten the 21 triples into a 63-value vector. Train a `63 → 32 → 3` network with a ReLU hidden layer, mean squared error, and Adam at `0.001` for 400 epochs. The model has 2,147 trainable parameters. |
| Live view and drawing | GLFW, PyOpenGL | Run inference on each frame with valid landmarks. Draw the prediction as a sphere and project it along the three axes with dotted guides. Store vertices and edges in box coordinates so the sketch stays in place when the view pans or rotates. |

I kept the landmarks in image coordinates so the network can use where the hand appears and how much of the frame it occupies. A hand can also look smaller when it turns or curls, which makes distance from a single camera ambiguous.

The box is centered at the origin, with X across the desk, Y away from me, and Z up. Its side lengths follow a 2:2:1 ratio in arbitrary units. The training labels include my aiming error. To measure how accurately the model places my fingertip, I’d need test positions with known physical coordinates.
