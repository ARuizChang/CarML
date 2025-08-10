import pygame
import math
import sys

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
GREY = (50, 50, 50)
BLUE = (0, 150, 255)
RED = (255, 0, 0)

# Track boundaries (simple rectangular circuit)
TRACK_WALLS = [
    pygame.Rect(100, 100, 600, 10),  # Top
    pygame.Rect(100, 490, 600, 10),  # Bottom
    pygame.Rect(100, 100, 10, 400),  # Left
    pygame.Rect(690, 100, 10, 400),  # Right
]


class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.length = 40
        self.width = 20
        self.max_speed = 5
        self.acceleration = 0.2
        self.turn_speed = 4
        self.ray_length = 100

    def update(self, keys):
        # Movement
        if keys[pygame.K_UP]:
            self.speed += self.acceleration
        elif keys[pygame.K_DOWN]:
            self.speed -= self.acceleration
        else:
            self.speed *= 0.95  # Friction

        self.speed = max(-self.max_speed, min(self.speed, self.max_speed))

        if keys[pygame.K_LEFT]:
            self.angle += self.turn_speed * (self.speed / self.max_speed)
        if keys[pygame.K_RIGHT]:
            self.angle -= self.turn_speed * (self.speed / self.max_speed)

        # Update position
        rad = math.radians(self.angle)
        self.x += -self.speed * math.sin(rad)
        self.y += -self.speed * math.cos(rad)

    def get_rect(self):
        return pygame.Rect(self.x - self.width // 2, self.y - self.length // 2, self.width, self.length)

    def draw(self, screen):
        # Draw the car as a rotated rectangle
        car_rect = pygame.Surface((self.width, self.length))
        car_rect.fill(RED)
        rotated = pygame.transform.rotate(car_rect, self.angle)
        rect = rotated.get_rect(center=(self.x, self.y))
        screen.blit(rotated, rect)

        # Draw rays
        self.draw_rays(screen)

    def draw_rays(self, screen):
        directions = [0, 45, 90, 135, 180, 225, 270, 315]  # Degrees relative to car
        self.ray_distances = []
        for d in directions:
            total_angle = (self.angle + d) % 360
            rad = math.radians(total_angle)
            for length in range(0, self.ray_length, 5):
                test_x = self.x + length * -math.sin(rad)
                test_y = self.y + length * -math.cos(rad)

                if self.check_collision_point((test_x, test_y)):
                    pygame.draw.line(screen, BLUE, (self.x, self.y), (test_x, test_y), 1)
                    self.ray_distances.append(length)
                    break
            else:
                pygame.draw.line(screen, BLUE, (self.x, self.y),
                                 (self.x + self.ray_length * -math.sin(rad),
                                  self.y + self.ray_length * -math.cos(rad)), 1)
                self.ray_distances.append(self.ray_length)

    def check_collision(self):
        rect = self.get_rect()
        for wall in TRACK_WALLS:
            if rect.colliderect(wall):
                return True
        return False

    def check_collision_point(self, point):
        px, py = point
        point_rect = pygame.Rect(px, py, 2, 2)
        for wall in TRACK_WALLS:
            if wall.colliderect(point_rect):
                return True
        return False

    def get_info(self):
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "angle": round(self.angle % 360, 2),
            "speed": round(self.speed, 2),
            "rays": self.ray_distances
        }


def draw_track(screen):
    for wall in TRACK_WALLS:
        pygame.draw.rect(screen, GREY, wall)


def draw_info(screen, car):
    font = pygame.font.SysFont(None, 24)
    info = car.get_info()
    y = 10
    for key, value in info.items():
        text = font.render(f"{key}: {value}", True, (255, 255, 255))
        screen.blit(text, (10, y))
        y += 20


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Driving Game")
    clock = pygame.time.Clock()

    car = Car(400, 300)

    running = True
    while running:
        clock.tick(FPS)
        screen.fill((0, 0, 0))

        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        car.update(keys)

        if car.check_collision():
            print("Collision!")

        # Draw
        draw_track(screen)
        car.draw(screen)
        draw_info(screen, car)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

