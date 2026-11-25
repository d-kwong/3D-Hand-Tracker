from math import radians, tau

import glfw
from OpenGL import GL

from collect_data import draw_live_views, open_live_views
from predict_live import main, predict_position
from target_box import DEFAULT_ORBIT, VIEW_HALF_SPAN
from webcam_landmarks import annotate_frame


def draw_sketch(vertices, edges):
    GL.glColor3f(1.0, 0.73, 0.28)
    GL.glBegin(GL.GL_LINES)
    for start, end in edges:
        GL.glVertex3f(*vertices[start])
        GL.glVertex3f(*vertices[end])
    GL.glEnd()

    GL.glPointSize(6)
    GL.glBegin(GL.GL_POINTS)
    for vertex in vertices:
        GL.glVertex3f(*vertex)
    GL.glEnd()


def run_drawing(model):
    vertices = []
    edges = []
    position = None
    pan_x = pan_y = 0.0
    azimuth, elevation = DEFAULT_ORBIT
    drag_button = None
    drag_start = None

    def on_key(window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_C:
            vertices.clear()
            edges.clear()
        elif key in (glfw.KEY_1, glfw.KEY_2) and position is not None:
            if key == glfw.KEY_2 and vertices:
                edges.append((len(vertices) - 1, len(vertices)))
            vertices.append(tuple(position))

    def on_mouse_button(window, button, action, mods):
        nonlocal drag_button, drag_start
        if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            return
        if action == glfw.PRESS:
            drag_button = button
            drag_start = glfw.get_cursor_pos(window)
        elif button == drag_button:
            drag_button = drag_start = None

    def on_cursor(window, x, y):
        nonlocal drag_start, pan_x, pan_y, azimuth, elevation
        if drag_start is None:
            return
        previous_x, previous_y = drag_start
        drag_start = (x, y)
        dx, dy = x - previous_x, y - previous_y
        if drag_button == glfw.MOUSE_BUTTON_RIGHT:
            azimuth = (azimuth - radians(0.35 * dx)) % tau
            elevation_limit = radians(85)
            elevation = max(-elevation_limit, min(elevation_limit, elevation + radians(0.35 * dy)))
            return
        short_side = min(glfw.get_window_size(window))
        if short_side > 0:
            units_per_pixel = 2 * VIEW_HALF_SPAN / short_side
            pan_x += dx * units_per_pixel
            pan_y -= dy * units_per_pixel

    def draw_geometry():
        draw_sketch(vertices, edges)

    print("1: place vertex. 2: place and connect. C: clear. "
          "Left-drag to pan. Right-drag to rotate.", flush=True)
    with open_live_views("Live Fingertip Drawing") as (box, sphere, preview, landmarker, camera):
        glfw.set_key_callback(box, on_key)
        glfw.set_key_callback(preview, on_key)
        glfw.set_mouse_button_callback(box, on_mouse_button)
        glfw.set_cursor_pos_callback(box, on_cursor)
        while not (glfw.window_should_close(box) or glfw.window_should_close(preview)):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            frame, hands = annotate_frame(frame, landmarker)
            position = predict_position(model, hands)
            draw_live_views(box, sphere, preview, frame, position,
                            pan=(pan_x, pan_y), orbit=(azimuth, elevation), draw_extra=draw_geometry)
            glfw.poll_events()


if __name__ == "__main__":
    raise SystemExit(main(run_drawing, "Draw with live fingertip predictions."))
