from contextlib import contextmanager
from math import atan2, cos, hypot, sin
from random import uniform

import glfw
from OpenGL import GL, GLU


# X runs across the desk, Y runs away from you, and Z points up.
# The origin is the center of the box. Coordinates use arbitrary units.
HALF_SIZE = (1.0, 1.0, 0.5)
TARGET_RADIUS = 0.045
VIEW_HALF_SPAN = 1.65
VIEW_DISTANCE = hypot(3, 5, 3)
DEFAULT_ORBIT = (atan2(-5, 3), atan2(3, hypot(3, 5)))
CORNERS = (
    (-1, -1, -0.5), (1, -1, -0.5), (1, 1, -0.5), (-1, 1, -0.5),
    (-1, -1, 0.5), (1, -1, 0.5), (1, 1, 0.5), (-1, 1, 0.5),
)
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def random_target():
    # Leave enough clearance for the full sphere to stay inside the box.
    return tuple(
        uniform(-extent + TARGET_RADIUS, extent - TARGET_RADIUS)
        for extent in HALF_SIZE
    )


def draw_scene(width, height, target, sphere, *, pan=(0, 0), orbit=DEFAULT_ORBIT,
               draw_extra=None):
    GL.glViewport(0, 0, width, height)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    aspect = width / height
    horizontal = VIEW_HALF_SPAN * max(1, aspect)
    vertical = VIEW_HALF_SPAN * max(1, 1 / aspect)
    pan_x, pan_y = pan
    GL.glOrtho(-horizontal - pan_x, horizontal - pan_x,
               -vertical - pan_y, vertical - pan_y, 0.1, 20)

    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    azimuth, elevation = orbit
    horizontal_distance = VIEW_DISTANCE * cos(elevation)
    eye_x = horizontal_distance * cos(azimuth)
    eye_y = horizontal_distance * sin(azimuth)
    eye_z = VIEW_DISTANCE * sin(elevation)
    GLU.gluLookAt(eye_x, eye_y, eye_z, 0, 0, 0, 0, 0, 1)

    GL.glColor3f(1, 1, 1)
    GL.glBegin(GL.GL_LINES)
    for start, end in EDGES:
        GL.glVertex3f(*CORNERS[start])
        GL.glVertex3f(*CORNERS[end])
    GL.glEnd()

    if draw_extra is not None:
        draw_extra()

    if target is None:
        return

    # Project the target straight onto each face.
    GL.glColor3f(0.55, 0.55, 0.55)
    GL.glEnable(GL.GL_LINE_STIPPLE)
    GL.glLineStipple(2, 0x1111)
    GL.glBegin(GL.GL_LINES)
    for axis, extent in enumerate(HALF_SIZE):
        face_point = list(target)
        for face in (-extent, extent):
            face_point[axis] = face
            GL.glVertex3f(*target)
            GL.glVertex3f(*face_point)
    GL.glEnd()
    GL.glDisable(GL.GL_LINE_STIPPLE)

    GL.glEnable(GL.GL_LIGHTING)
    GL.glColor3f(0.03, 0.75, 0.82)
    GL.glTranslatef(*target)
    GLU.gluSphere(sphere, TARGET_RADIUS, 24, 16)
    GL.glDisable(GL.GL_LIGHTING)


@contextmanager
def open_target_box(title="Target Box"):
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW.")

    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        window = glfw.create_window(800, 600, title, None, None)
        if not window:
            raise RuntimeError("Could not create an OpenGL window.")

        glfw.make_context_current(window)
        glfw.swap_interval(1)
        GL.glClearColor(0, 0, 0, 1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (-2, 3, 4, 0))

        sphere = GLU.gluNewQuadric()
        try:
            yield window, sphere
        finally:
            GLU.gluDeleteQuadric(sphere)
    finally:
        glfw.terminate()


def main():
    with open_target_box() as (window, sphere):
        target = random_target()

        def on_key(window, key, scancode, action, mods):
            nonlocal target
            if action != glfw.PRESS:
                return
            if key == glfw.KEY_SPACE:
                x, y, z = target
                print(f"Mock save: x={x:.3f}, y={y:.3f}, z={z:.3f}", flush=True)
                target = random_target()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

        glfw.set_key_callback(window, on_key)
        while not glfw.window_should_close(window):
            width, height = glfw.get_framebuffer_size(window)
            if width and height:
                draw_scene(width, height, target, sphere)
                glfw.swap_buffers(window)
            glfw.wait_events()


if __name__ == "__main__":
    main()
