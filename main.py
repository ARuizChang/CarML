import math
import pygame
from pygame.math import Vector2
import neat
import os
import pickle
import argparse
import copy

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
    def ccw(p1, p2, p3):
        return (p3.y - p1.y)*(p2.x - p1.x) > (p2.y - p1.y)*(p3.x - p1.x)
    return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))

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

def cast_ray(origin, direction, segments, max_length):
    nearest_t, nearest_point = None, None
    for (a, b) in segments:
        t, point = seg_intersection(origin, direction, a, b - a)
        if t is not None and 0 <= t <= max_length:
            if nearest_t is None or t < nearest_t:
                nearest_t, nearest_point = t, point
    return nearest_t, nearest_point

class Car:
    def __init__(self, pos):
        self.pos = Vector2(pos)
        self.heading = 0.0
        self.velocity = 0.0
        self.current_checkpoint = 0
        self.last_pos = Vector2(pos)
        self.passed_checkpoints = set()  # Track passed checkpoints in this lap

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
        self.heading %= 360

        rad = math.radians(self.heading)
        forward = Vector2(math.cos(rad), math.sin(rad))
        self.pos += forward * self.velocity * dt

    def rays(self, count=8, length=RAY_LENGTH):
        angles = [self.heading + (i * 360 / count) for i in range(count)]
        return [(Vector2(math.cos(math.radians(a)), math.sin(math.radians(a))), length) for a in angles]

def update_checkpoint_progress(car):
    cp_start, cp_end = checkpoints[car.current_checkpoint]
    crossed = lines_intersect(car.last_pos, car.pos, cp_start, cp_end)
    if crossed:
        car.current_checkpoint = (car.current_checkpoint + 1) % len(checkpoints)
        return True
    return False

def calculate_reward(car, ray_distances, time_step_penalty=-0.05):
    filtered_dists = [d for d in ray_distances if d is not None]
    min_dist = min(filtered_dists) if filtered_dists else RAY_LENGTH

    if min_dist < 10:
        return -10.0  # Strong collision penalty

    passed_checkpoint = update_checkpoint_progress(car)
    reward_checkpoint = 0.0
    reward_finish = 0.0

    # Only reward for new checkpoints
    if passed_checkpoint:
        if car.current_checkpoint not in car.passed_checkpoints:
            car.passed_checkpoints.add(car.current_checkpoint)
            reward_checkpoint = 10.0

        # If all checkpoints passed, give big bonus and reset for next lap
        if len(car.passed_checkpoints) == len(checkpoints):
            reward_finish = 100.0
            car.passed_checkpoints.clear()

    reward_velocity = max(car.velocity / MAX_SPEED, 0.0) * 0.5
    reward_time = time_step_penalty

    # Penalize for going backwards
    backward_penalty = -2.0 if car.velocity < -5 else 0.0

    reward = reward_checkpoint + reward_finish + reward_velocity + reward_time + backward_penalty
    return reward

def normalize_ray_distances(ray_distances):
    normalized = []
    for d in ray_distances:
        if d is None:
            normalized.append(1.0)
        else:
            normalized.append(min(d / RAY_LENGTH, 1.0))
    return normalized

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

def eval_genomes(genomes, config):
    nets = []
    cars = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        car = Car((100, 400))
        car.heading = 270
        cars.append(car)
        ge.append(genome)

    max_frames = 1000
    frame = 0
    run = True

    while run and len(cars) > 0 and frame < max_frames:
        dt = clock.tick(FPS) / 1000.0
        frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        remove_indices = []
        for i, car in enumerate(cars):
            ray_data = []
            for dir_vec, length in car.rays(8, RAY_LENGTH):
                dist, _ = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
                ray_data.append(dist)

            inputs = normalize_ray_distances(ray_data)
            norm_velocity = (car.velocity + MAX_SPEED * 0.5) / (MAX_SPEED * 1.5)
            norm_heading = car.heading / 360.0
            inputs.append(norm_velocity)
            inputs.append(norm_heading)

            output = nets[i].activate(inputs)

            accel_input = (output[0] * 2) - 1
            steer_input = (output[1] * 2) - 1
            handbrake = output[2] > 0.5

            car.update(dt, accel_input, steer_input, handbrake)
            reward = calculate_reward(car, ray_data)
            ge[i].fitness += reward

            if reward < -0.5:
                ge[i].fitness -= 5
                remove_indices.append(i)

        for i in reversed(remove_indices):
            cars.pop(i)
            nets.pop(i)
            ge.pop(i)

        if len(cars) > 0:
            screen.fill(BG)
            for a, b in track_segments:
                pygame.draw.line(screen, TRACK_COLOR, a, b, 4)

            for i_cp, (start, end) in enumerate(checkpoints):
                color = (0, 255, 0) if i_cp == cars[0].current_checkpoint else (150, 150, 150)
                pygame.draw.line(screen, color, start, end, 2)

            car = cars[0]
            pygame.draw.polygon(screen, CAR_COLOR, car_polygon(car.pos, car.heading))

            for dir_vec, length in car.rays(8, RAY_LENGTH):
                dist, point = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
                end_point = point if point is not None else car.pos + dir_vec * length
                pygame.draw.line(screen, RAY_COLOR, car.pos, end_point, 1)
                if point:
                    pygame.draw.circle(screen, (255, 0, 0), (int(point.x), int(point.y)), 4)

            reward_display = calculate_reward(car, ray_data)
            draw_hud(screen, ray_data, car.velocity, car.heading, reward_display, car.current_checkpoint)
            pygame.display.flip()

def car_polygon(pos, heading):
    w, h = CAR_SIZE
    rad = math.radians(heading)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    corners = [
        Vector2(-w/2, -h/2),
        Vector2(w/2, -h/2),
        Vector2(w/2, h/2),
        Vector2(-w/2, h/2)
    ]
    rotated = [Vector2(c.x * cos_a - c.y * sin_a, c.x * sin_a + c.y * cos_a) + pos for c in corners]
    return rotated

def run_neat(config_file, restore_agent=None, generations=20):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # If restoring from agent, set all population to a deep copy of that genome with correct keys and fitness
    if restore_agent is not None:
        with open(restore_agent, "rb") as f:
            winner = pickle.load(f)
        for gid in p.population:
            g = copy.deepcopy(winner)
            g.key = gid
            g.fitness = 0.0  # Ensure fitness is set!
            p.population[gid] = g
        # Re-speciate the population after restoring
        p.species.speciate(config, p.population, p.generation)

    winner = p.run(eval_genomes, generations)

    with open("best_car.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome saved as best_car.pkl")

def play_human():
    car = Car((100, 400))
    car.heading = 270
    run = True
    while run:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        keys = pygame.key.get_pressed()
        accel = 0
        steer = 0
        handbrake = False
        if keys[pygame.K_UP]:
            accel = 1
        elif keys[pygame.K_DOWN]:
            accel = -1
        if keys[pygame.K_LEFT]:
            steer = -1
        elif keys[pygame.K_RIGHT]:
            steer = 1
        if keys[pygame.K_SPACE]:
            handbrake = True

        car.update(dt, accel, steer, handbrake)
        ray_data = []
        for dir_vec, length in car.rays(8, RAY_LENGTH):
            dist, _ = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
            ray_data.append(dist)

        screen.fill(BG)
        for a, b in track_segments:
            pygame.draw.line(screen, TRACK_COLOR, a, b, 4)
        for i_cp, (start, end) in enumerate(checkpoints):
            color = (0, 255, 0) if i_cp == car.current_checkpoint else (150, 150, 150)
            pygame.draw.line(screen, color, start, end, 2)
        pygame.draw.polygon(screen, CAR_COLOR, car_polygon(car.pos, car.heading))
        for dir_vec, length in car.rays(8, RAY_LENGTH):
            dist, point = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
            end_point = point if point is not None else car.pos + dir_vec * length
            pygame.draw.line(screen, RAY_COLOR, car.pos, end_point, 1)
            if point:
                pygame.draw.circle(screen, (255, 0, 0), (int(point.x), int(point.y)), 4)
        reward_display = calculate_reward(car, ray_data)
        draw_hud(screen, ray_data, car.velocity, car.heading, reward_display, car.current_checkpoint)
        pygame.display.flip()

def view_agent(pkl_file, config_file):
    with open(pkl_file, "rb") as f:
        genome = pickle.load(f)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    car = Car((100, 400))
    car.heading = 270
    run = True
    while run:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        ray_data = []
        for dir_vec, length in car.rays(8, RAY_LENGTH):
            dist, _ = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
            ray_data.append(dist)
        inputs = normalize_ray_distances(ray_data)
        norm_velocity = (car.velocity + MAX_SPEED * 0.5) / (MAX_SPEED * 1.5)
        norm_heading = car.heading / 360.0
        inputs.append(norm_velocity)
        inputs.append(norm_heading)
        output = net.activate(inputs)
        accel_input = (output[0] * 2) - 1
        steer_input = (output[1] * 2) - 1
        handbrake = output[2] > 0.5
        car.update(dt, accel_input, steer_input, handbrake)

        screen.fill(BG)
        for a, b in track_segments:
            pygame.draw.line(screen, TRACK_COLOR, a, b, 4)
        for i_cp, (start, end) in enumerate(checkpoints):
            color = (0, 255, 0) if i_cp == car.current_checkpoint else (150, 150, 150)
            pygame.draw.line(screen, color, start, end, 2)
        pygame.draw.polygon(screen, CAR_COLOR, car_polygon(car.pos, car.heading))
        for dir_vec, length in car.rays(8, RAY_LENGTH):
            dist, point = cast_ray(car.pos, dir_vec, track_segments, RAY_LENGTH)
            end_point = point if point is not None else car.pos + dir_vec * length
            pygame.draw.line(screen, RAY_COLOR, car.pos, end_point, 1)
            if point:
                pygame.draw.circle(screen, (255, 0, 0), (int(point.x), int(point.y)), 4)
        reward_display = calculate_reward(car, ray_data)
        draw_hud(screen, ray_data, car.velocity, car.heading, reward_display, car.current_checkpoint)
        pygame.display.flip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "view", "play"], default="train", help="Mode: train, view, play")
    parser.add_argument("--agent", type=str, help="Path to .pkl agent file (for view or resume training)")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations to train")
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    if args.mode == "train":
        run_neat(config_path, restore_agent=args.agent, generations=args.generations)
    elif args.mode == "view":
        if not args.agent:
            print("Please provide --agent path to .pkl file.")
        else:
            view_agent(args.agent, config_path)
    elif args.mode == "play":
        play_human()