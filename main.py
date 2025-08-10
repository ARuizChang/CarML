import math
import pygame
from pygame.math import Vector2

WIDTH, HEIGHT = 1024, 720
FPS = 60

CAR_SIZE = (36, 18)
MAX_SPEED = 400.0
ACCELERATION = 600.0
FRICTION = 800.0
STEER_SPEED = 160.0
RAY_LENGTH = 800

BG = (30, 30, 30)
TRACK_COLOR = (200, 200, 200)
CAR_COLOR = (50, 180, 50)
RAY_COLOR = (220, 50, 50)
HUD_BG = (20, 20, 20)
HUD_TEXT = (230, 230, 230)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

track_segments = []

def add_rect_outline(x, y, w, h):
    corners = [Vector2(x, y), Vector2(x + w, y), Vector2(x + w, y + h), Vector2(x, y + h)]
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        track_segments.append((a, b))

add_rect_outline(40, 40, WIDTH - 80, HEIGHT - 120)
add_rect_outline(300, 180, 420, 280)
track_segments.append((Vector2(140, 500), Vector2(300, 420)))
track_segments.append((Vector2(740, 420), Vector2(880, 500)))

def seg_intersection(p, r, q, s):
    r_cross_s = r.cross(s)
    q_p = q - p
    if abs(r_cross_s) < 1e-8:
        return None, None
    t = q_p.cross(s) / r_cross_s
    u = q_p.cross(r) / r_cross_s
    if t >= 0 and 0 <= u <= 1:
        point = p + r * t
        return t, point
    return None, None

class Car:
    def __init__(self, pos):
        self.pos = Vector2(pos)
        self.heading = 0.0
        self.velocity = 0.0

    def update(self, dt, accel_input, steer_input, handbrake=False):
        if accel_input != 0:
            self.velocity += ACCELERATION * accel_input * dt
        else:
            if self.velocity > 0:
                self.velocity -= FRICTION * dt
                if self.velocity < 0:
                    self.velocity = 0
            elif self.velocity < 0:
                self.velocity += FRICTION * dt
                if self.velocity > 0:
                    self.velocity = 0

        self.velocity = max(min(self.velocity, MAX_SPEED), -MAX_SPEED * 0.5)

        if handbrake:
            if self.velocity > 0:
                self.velocity -= FRICTION * 3 * dt
                if self.velocity < 0:
                    self.velocity = 0
            elif self.velocity < 0:
                self.velocity += FRICTION * 3 * dt
                if self.velocity > 0:
                    self.velocity = 0

        steer_effect = abs(self.velocity) / MAX_SPEED
        steer_angle = STEER_SPEED * steer_input * (steer_effect if steer_effect > 0.05 else 0.05) * dt
        self.heading += steer_angle
        self.heading %= 360  # keep angle between 0 and 360

        rad = math.radians(self.heading)
        forward = Vector2(math.cos(rad), math.sin(rad))
        self.pos += forward * self.velocity * dt

    def get_corners(self):
        w, h = CAR_SIZE
        hw, hh = w / 2, h / 2
        rad = math.radians(self.heading)
        cos, sin = math.cos(rad), math.sin(rad)
        local = [Vector2(-hw, -hh), Vector2(hw, -hh), Vector2(hw, hh), Vector2(-hw, hh)]
        return [Vector2(v.x * cos - v.y * sin, v.x * sin + v.y * cos) + self.pos for v in local]

    def rays(self, count=8, length=RAY_LENGTH):
        angles = [self.heading + (i * 360 / count) for i in range(count)]
        return [(Vector2(math.cos(math.radians(a)), math.sin(math.radians(a))), length) for a in angles]

def cast_ray(origin, direction, segments, max_length):
    nearest_t, nearest_point = None, None
    for (a, b) in segments:
        t, point = seg_intersection(origin, direction, a, b - a)
        if t is not None and 0 <= t <= max_length:
            if nearest_t is None or t < nearest_t:
                nearest_t, nearest_point = t, point
    return nearest_t, nearest_point

car = Car((200, 200))

def draw_hud(surface, distances, velocity, angle_deg):
    # Group ray distances into two lines
    ray_strs = [f"{d:.1f}" if d is not None else "none" for d in distances]
    line1 = "Rays: " + ", ".join(ray_strs[:4])
    line2 = "      " + ", ".join(ray_strs[4:])

    lines = [line1, line2, f"Velocity: {velocity:.1f} px/s", f"Angle: {angle_deg:.1f} deg"]

    total_h = len(lines) * 22 + 16
    hud_x = 8
    hud_y = HEIGHT - total_h - 8

    # Semi-transparent HUD background
    hud_surface = pygame.Surface((320, total_h), pygame.SRCALPHA)
    hud_surface.fill((HUD_BG[0], HUD_BG[1], HUD_BG[2], 160))  # alpha 160
    surface.blit(hud_surface, (hud_x, hud_y))

    for i, text in enumerate(lines):
        surface.blit(font.render(text, True, HUD_TEXT), (hud_x + 8, hud_y + 8 + i * 22))

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    accel_input = steer_input = 0.0
    handbrake = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        accel_input = 1.0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        accel_input = -1.0

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer_input = -1.0
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer_input = 1.0

    if keys[pygame.K_SPACE]:
        handbrake = True

    car.update(dt, accel_input, steer_input, handbrake)

    screen.fill(BG)
    for a, b in track_segments:
        pygame.draw.line(screen, TRACK_COLOR, a, b, 4)

    ray_data = []
    for dir_vec, length in car.rays(8, RAY_LENGTH):
        dist, hit_point = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
        ray_data.append(dist)
        end_point = hit_point if hit_point else car.pos + dir_vec * RAY_LENGTH
        pygame.draw.line(screen, RAY_COLOR, car.pos, end_point, 2)
        pygame.draw.circle(screen, RAY_COLOR, (int(end_point.x), int(end_point.y)), 4)

    pygame.draw.polygon(screen, CAR_COLOR, car.get_corners())
    nose = sum((car.get_corners()[1], car.get_corners()[2]), Vector2()) / 2
    pygame.draw.circle(screen, (10, 10, 10), (int(nose.x), int(nose.y)), 4)

    draw_hud(screen, ray_data, car.velocity, car.heading)

    pygame.display.flip()

pygame.quit()
