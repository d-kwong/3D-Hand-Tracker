# 3D Hand Tracker

In this project, I wanted to do a quick proof of concept for an idea i had. Could I draw in 3D just by moving my index finger in front of a webcam? I wanted to see how far I could get with one camera and a small model.

I got the hand landmarks working with MediaPipe’s open-source model. They showed where my fingertip was in the image and how far forward it was relative to my wrist. When I moved my whole hand closer to the webcam, my wrist moved with it. That was the part I wanted to figure out for drawing. Where was my fingertip in the space in front of the camera?

To get training data, I put a box on screen and had the app show me one target at a time. I pointed where I imagined each target would be in front of me and pressed Space to save my hand landmarks with the target’s coordinates. That meant pointing at about 100 imaginary spots before I could train the PyTorch model and try drawing with it.

<img src="docs/media/draw-and-rotate.gif" alt="Mirrored hand landmarks alongside a synchronized 3D view, where the estimated fingertip places amber vertices before the finished sketch rotates" width="960">

*On the left is my hand with the MediaPipe landmarks. On the right, the teal dot follows the model’s fingertip estimate. I place vertices and edges, then rotate the finished sketch to see its depth.*

## Live pipeline

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/pipeline-outline-dark.png">
  <img src="docs/media/pipeline-outline-light.png" alt="OpenCV captures a webcam frame. MediaPipe extracts 21 landmark triples. A 63 to 32 to 3 PyTorch model predicts the fingertip position, and OpenGL displays it in a 3D workspace." width="960">
</picture>

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

The box is centered at the origin, with X across the desk, Y away from me, and Z up. Its side lengths follow a 2:2:1 ratio in arbitrary units.

## What I’d try next

The first thing I’d improve is the training data. Pointing at an imagined target leaves my aiming error in the label. I’d try tracking colored markers with two synchronized, calibrated cameras and triangulating their positions. That setup would be used for data collection and evaluation, while the trained model would still take a single unmarked camera view at inference.

I’d also like to compare single-camera and multi-camera tracking, then extend the idea from fingertip position to full-body pose. The longer-term experiment would be to see how much 3D body pose I could recover from an ordinary phone video.
