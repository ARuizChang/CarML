import math
import pygame
from pygame.math import Vector2

WIDTH, HEIGHT = 1024, 720
FPS = 60

CAR_SIZE = (36, 18)
MAX_SPEED = 600.0
ACCELERATION = 600.0
FRICTION = 400.0
STEER_SPEED = 160.0
RAY_LENGTH = 800

BG = (30, 30, 30)
TRACK_COLOR = (200, 200, 200)
CAR_COLOR = (50, 180, 50)
RAY_COLOR = (220, 220, 0)
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
track_segments.append((Vector2(700, 420), Vector2(880, 500)))

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

# Define checkpoints as line segments crossing the track
checkpoints = [
    (Vector2(0, 350), Vector2(350, 350)),
    (Vector2(0, 250), Vector2(350, 250)),
    (Vector2(0, 150), Vector2(350, 185)),
    (Vector2(0, 0), Vector2(350, 185)),
    (Vector2(350, 0), Vector2(350, 185)),
    (Vector2(450, 0), Vector2(450, 185)),
    (Vector2(550, 0), Vector2(550, 185)),
    (Vector2(650, 0), Vector2(650, 185)),
    (Vector2(750, 0), Vector2(700, 185)),
    (Vector2(1000, 0), Vector2(720, 185)),
    (Vector2(1000, 150), Vector2(720, 185)),
    (Vector2(1000, 300), Vector2(720, 300)),
    (Vector2(1000, 450), Vector2(720, 450)),
    (Vector2(1000, 600), Vector2(720, 450)),
    (Vector2(650, 700), Vector2(650, 450)),
    (Vector2(550, 700), Vector2(550, 450)),
    (Vector2(450, 700), Vector2(450, 450)),
    (Vector2(350, 700), Vector2(350, 450)),
    (Vector2(250, 700), Vector2(250, 450)),
    (Vector2(0, 700), Vector2(150, 500)),
    (Vector2(0, 450), Vector2(250, 450)),
]


def lines_intersect(a1, a2, b1, b2):
    # Check if line segments a1a2 and b1b2 intersect
    def ccw(p1, p2, p3):
        return (p3.y - p1.y)*(p2.x - p1.x) > (p2.y - p1.y)*(p3.x - p1.x)
    return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))

class Car:
    def __init__(self, pos):
        self.pos = Vector2(pos)
        self.heading = 0.0
        self.velocity = 0.0
        self.current_checkpoint = 0
        self.last_pos = Vector2(pos)

    def update(self, dt, accel_input, steer_input, handbrake=False):
        self.last_pos = Vector2(self.pos)

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

def update_checkpoint_progress(car):
    # Check if line from last_pos to current pos crosses current checkpoint line segment
    cp_start, cp_end = checkpoints[car.current_checkpoint]
    crossed = lines_intersect(car.last_pos, car.pos, cp_start, cp_end)

    if crossed:
        car.current_checkpoint = (car.current_checkpoint + 1) % len(checkpoints)
        return True
    return False

def calculate_reward(car, ray_distances, time_step_penalty=-0.01):
    filtered_dists = [d for d in ray_distances if d is not None]
    min_dist = min(filtered_dists) if filtered_dists else RAY_LENGTH

    if min_dist < 10:
        return -1.0  # crash penalty

    passed_checkpoint = update_checkpoint_progress(car)
    reward_checkpoint = 1.0 if passed_checkpoint else 0.0

    # Reward velocity forward
    reward_velocity = max(car.velocity / MAX_SPEED, 0)

    reward_time = time_step_penalty

    reward = 2.0 * reward_checkpoint + 0.3 * reward_velocity + reward_time

    return reward

def draw_hud(surface, distances, velocity, angle_deg, reward, checkpoint_idx):
    ray_strs = [f"{d:.1f}" if d is not None else "none" for d in distances]
    line1 = "Rays: " + ", ".join(ray_strs[:4])
    line2 = "      " + ", ".join(ray_strs[4:])

    lines = [
        line1,
        line2,
        f"Velocity: {velocity:.1f} px/s",
        f"Angle: {angle_deg:.1f} deg",
        f"Reward: {reward:.3f}",
        f"Checkpoint: {checkpoint_idx + 1}/{len(checkpoints)}"
    ]

    total_h = len(lines) * 22 + 16
    hud_x = 8
    hud_y = HEIGHT - total_h - 8

    hud_surface = pygame.Surface((320, total_h), pygame.SRCALPHA)
    hud_surface.fill((HUD_BG[0], HUD_BG[1], HUD_BG[2], 160))
    surface.blit(hud_surface, (hud_x, hud_y))

    for i, text in enumerate(lines):
        surface.blit(font.render(text, True, HUD_TEXT), (hud_x + 8, hud_y + 8 + i * 22))

car = Car((100, 400))
car.heading = 270
total_reward = 0.0

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

    ray_data = []
    for dir_vec, length in car.rays(8, RAY_LENGTH):
        dist, hit_point = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
        ray_data.append(dist)

    reward = calculate_reward(car, ray_data)
    total_reward += reward

    if reward < -0.5:
        car = Car((100, 400))
        car.heading = 270
        total_reward = 0.0

    screen.fill(BG)
    for a, b in track_segments:
        pygame.draw.line(screen, TRACK_COLOR, a, b, 4)

    # Draw checkpoints as red lines (thicker)
    for i, (start, end) in enumerate(checkpoints):
        color = (0, 255, 0) if i == car.current_checkpoint else (200, 50, 50)
        pygame.draw.line(screen, color, start, end, 4)

    for dir_vec, length in car.rays(8, RAY_LENGTH):
        dist, hit_point = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
        end_point = hit_point if hit_point else car.pos + dir_vec * RAY_LENGTH
        pygame.draw.line(screen, RAY_COLOR, car.pos, end_point, 2)
        pygame.draw.circle(screen, RAY_COLOR, (int(end_point.x), int(end_point.y)), 4)

    pygame.draw.polygon(screen, CAR_COLOR, car.get_corners())
    nose = sum((car.get_corners()[1], car.get_corners()[2]), Vector2()) / 2
    pygame.draw.circle(screen, (10, 10, 10), (int(nose.x), int(nose.y)), 4)

    draw_hud(screen, ray_data, car.velocity, car.heading, reward, car.current_checkpoint)

    pygame.display.flip()

pygame.quit()
