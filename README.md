# Interactive-projector

# Runtime Architecture — Contributor Guide

## Overview

The runtime has two threads talking through a shared queue:

```
[Vision Thread]  →  [Bounded Queue]  →  [Game Loop]  →  [Active Mode]
  CV/camera             queue_bus            game_loop        modes/
```

You should only ever need to touch **one layer**. Find your team below.

---

## Important: what MediaPipe actually gives you

This is worth understanding before you write any code.

MediaPipe does **not** automatically generate JSON or events. What it does is:
every time you feed it a video frame, it returns landmark positions — coordinates
for each joint on a detected hand — in normalised 0-1 values (0,0 = top-left of
frame, 1,1 = bottom-right). That's it. You then write logic on top of those
coordinates to decide "is this a tap? a drag?" and post a GameEvent yourself.

The full flow on every camera frame:

```
camera captures frame
        ↓
OpenCV reads it as a pixel array
        ↓
you pass it to MediaPipe
        ↓
MediaPipe returns landmark positions in 0-1 coords
        ↓
you multiply by frame width/height → actual pixel coords
        ↓
you apply your own logic (is index finger close to thumb? → TAP)
        ↓
you create a GameEvent and post it to the queue
```

This all happens automatically in the `while` loop inside `run_vision()` —
~30 times per second as the camera produces frames. The CV team writes
the detection logic once, the loop runs it continuously.

---

## For the CV / Projection team → `vision_thread.py`

Your job: detect gestures from the camera and post `GameEvent` objects onto the queue.

### The contract
You must produce `GameEvent` objects from `events.py`. That's the only interface
between your work and the game loop — everything else is your business.

```python
from events import EventType, GameEvent
from queue_bus import event_queue
import queue

# Post an event like this:
try:
    event_queue.put_nowait(GameEvent(EventType.TAP, {"x": 320, "y": 240}))
except queue.Full:
    pass  # intentionally drop if game loop is lagging — never block here
```

### Event types
| EventType | When to fire | Required payload keys |
|---|---|---|
| `TAP` | Gesture/touch begins | `x`, `y` |
| `DRAG` | Gesture held and moving | `x`, `y` |
| `RELEASE` | Gesture ends | `x`, `y` (optional) |

`x` and `y` should be pixel coordinates relative to the **projected display**,
not the camera frame — so account for any projection offset/scaling in your transform.

### Where to put your code
Replace the contents of `run_vision()` in `vision_thread.py` with your detection loop.
Keep the function signature: `def run_vision(running_flag: threading.Event) -> None`
Check `running_flag.is_set()` in your while loop — this is how shutdown works cleanly.

### Adding new gesture types
If your CV work produces gestures the current events don't cover (e.g. `SWIPE`, `PINCH_ZOOM`):
1. Add the new type to `EventType` in `events.py`
2. Tell the game/UI team what payload keys it carries
3. They handle it in their mode's `handle_event`

### Integrating your calibration matrix
If you ran a calibration script to fix parallax between the camera and projector,
it produced a homography matrix — the output of `cv2.findHomography()`. Do not
recompute it in `vision_thread.py`. Instead:

1. Save it from your calibration script: `np.save("calibration.npy", H)`
2. Place `calibration.npy` in the project root
3. `vision_thread.py` loads it automatically on startup and applies it to every
   coordinate before it becomes a `GameEvent`

If the file is missing, the vision thread will print a warning and pass coordinates
through unchanged — fine for testing, but parallax will be visible in production.

The transform is applied here in `run_vision()`:
```python
raw_x, raw_y = _landmark_to_screen(landmark, frame_w, frame_h)
x, y = _camera_to_projector(raw_x, raw_y)  # homography applied here
_post(GameEvent(EventType.TAP, {"x": x, "y": y}))
```
You do not need to touch any of this — just provide the `.npy` file.

### Latency mitigations (already applied)
The following are already set in `vision_thread.py` — do not revert them:

| Setting | What it does |
|---|---|
| `model_complexity=0` | Uses MediaPipe lite model — biggest single latency win |
| `CAP_PROP_BUFFERSIZE=1` | Always reads the latest camera frame, not a queued stale one |
| `640x480` resolution | Sufficient for hand tracking, much faster than 1080p |

If latency is still felt during user testing, the next thing to try is reducing
`min_detection_confidence` slightly (e.g. 0.6) to speed up detection.

### Mouse injection (remove when your code is in)
`game_loop.py` currently simulates events from mouse input so the game is testable
without a camera. Once your vision thread is producing real events, remove the
mouse injection block — it's clearly marked with a comment in `game_loop.py`.

---

## For the Game / UI team → `modes/`

Your job: build the game. The architecture is **not** static screens — a `GameMode`
is a fully running, stateful, interactive application that owns its own data and
redraws itself 60 times per second.

### The mental model
Think of it exactly like a Flutter `StatefulWidget`:
- `self.grid` / `self.score` / `self.turn` = your `State` class fields
- `handle_event()` = `setState(() { ... })` — mutate state, rendering follows automatically
- `render()` = `build()` — redraws everything from current state each frame
- `update()` = a `Ticker` or `AnimationController` callback — runs every frame regardless of input

**`render()` never accumulates — it redraws from scratch every frame.**
State lives in the class. Rendering is a pure function of that state.
Don't draw anything inside `handle_event` — just update state, and let `render` handle the visuals.

### The contract
Subclass `GameMode` from `modes/base.py` and implement three methods:

```python
from modes.base import GameMode
from events import EventType, GameEvent
import pygame

class BattleshipsMode(GameMode):

    def __init__(self):
        # ALL game state lives here
        self.grid = [[None] * 10 for _ in range(10)]
        self.current_turn = "player1"
        self.cell_size = 60

    def handle_event(self, event: GameEvent):
        # Called for every vision/input event this frame.
        # Update your state here — don't draw anything here.
        if event.type == EventType.TAP:
            col = event.payload["x"] // self.cell_size
            row = event.payload["y"] // self.cell_size
            self.grid[row][col] = "hit" if self._is_ship(row, col) else "miss"

        # Return a GameMode instance to switch to that mode:
        # e.g. return MenuMode() transitions immediately after this frame.

    def update(self):
        # Called once per frame.
        # Use for animations, turn timers, AI moves — anything time-based.
        pass

    def render(self, surface):
        # Called 60x per second. Redraws the entire game from current state.
        # surface is a pygame.Surface the size of the window — draw onto it.
        for row in range(10):
            for col in range(10):
                color = self._cell_color(row, col)
                rect = pygame.Rect(
                    col * self.cell_size, row * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 1)  # grid lines
```

### Porting from your Flutter app
If you have existing game logic from a Flutter build:
- **State fields** (`int score`, `List<List<Cell>> grid`, etc.) → move into `__init__` as Python equivalents. Direct 1:1 mapping.
- **`setState(() { ... })` blocks** → move into `handle_event`. Same logic, just Python syntax.
- **`build()` / widget tree** → rewrite as `pygame.draw` calls in `render()`. Your existing layout decisions (grid sizing, colours, padding, positions) all still apply — just express them as pixel coordinates instead of Flutter widgets.
- **`Ticker` / `AnimationController` callbacks** → move into `update()`.
- You do **not** need to redesign the game or rethink the logic. Only the rendering layer changes.

### Adding a new mode
1. Create `modes/your_mode.py` with your `GameMode` subclass
2. Import it at the top of `game_loop.py` (marked section)
3. Either set it as the starting mode, or return it from another mode's `handle_event`

### Switching between modes
Return a mode instance from `handle_event` to trigger a transition:

```python
def handle_event(self, event: GameEvent):
    if event.type == EventType.TAP and self.game_over:
        return MenuMode()  # game loop switches to this immediately
```

### Drawing reference
`surface` in `render()` is a standard `pygame.Surface`.
See `modes/draw_mode.py` for a worked example — it shows persistent canvas drawing,
handling TAP/DRAG/RELEASE, and the render pattern.

---

## File map — who touches what

```
main.py              — entry point, starts both threads.         DON'T TOUCH
events.py            — GameEvent + EventType definitions.        CV team adds types here
queue_bus.py         — the shared queue instance.                DON'T TOUCH
vision_thread.py     — camera + gesture detection loop.          CV TEAM EDITS HERE
game_loop.py         — mode registration, mouse injection.       MINIMAL EDITS ONLY
modes/
  base.py            — GameMode base class / interface.          DON'T TOUCH
  draw_mode.py       — reference implementation.                 READ, DON'T EDIT
  [your_mode].py     — your game modes go here.                  GAME/UI TEAM ADDS HERE
```

---

## Running

```bash
pip install pygame mediapipe opencv-python
python main.py
```

Press `Esc` to quit.
Mouse left-click drag simulates vision input until the CV team integrates.
