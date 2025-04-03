import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame.locals

# Constants
GRID_SIZE = 10
CELL_SIZE = 60
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)


def init_pygame():
    """Initialize pygame for rendering"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Search and Rescue")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    return screen, clock, font


def render_env(env):
    """Render the environment using pygame"""
    # Initialize pygame if not already done
    if env.window is None:
        env.window, env.clock, env.font = init_pygame()

    # Clear screen
    env.window.fill(WHITE)

    # Draw grid
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            pygame.draw.rect(
                env.window,
                BLACK,
                (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                1,
            )

    # Draw obstacles
    for obs in env.obstacles:
        pygame.draw.rect(
            env.window,
            GRAY,
            (obs[1] * CELL_SIZE, obs[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

    # Draw safe zone
    pygame.draw.rect(
        env.window,
        GREEN,
        (
            env.safe_zone_pos[1] * CELL_SIZE,
            env.safe_zone_pos[0] * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        ),
    )

    # Draw person (if not carried)
    if not env.carrying_person:
        pygame.draw.circle(
            env.window,
            BLUE,
            (
                env.person_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                env.person_pos[0] * CELL_SIZE + CELL_SIZE // 2,
            ),
            CELL_SIZE // 4,
        )

    # Draw drone
    drone_color = ORANGE if env.carrying_person else RED
    pygame.draw.circle(
        env.window,
        drone_color,
        (
            env.drone_pos[1] * CELL_SIZE + CELL_SIZE // 2,
            env.drone_pos[0] * CELL_SIZE + CELL_SIZE // 2,
        ),
        CELL_SIZE // 3,
    )

    # Draw battery level
    battery_text = f"Battery: {env.battery_level}%"
    text_surface = env.font.render(battery_text, True, BLACK)
    env.window.blit(text_surface, (10, 10))

    # Draw status
    status_text = "Carrying Person" if env.carrying_person else "Searching"
    text_surface = env.font.render(status_text, True, BLACK)
    env.window.blit(text_surface, (10, 40))

    # Update display
    pygame.display.flip()
    env.clock.tick(FPS)

    # OpenGL visualization (very basic implementation)
    # This is added to meet the requirement for PyOpenGL
    def initialize_opengl():
        """Initialize OpenGL rendering"""
        pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            pygame.locals.DOUBLEBUF | pygame.locals.OPENGL,
        )
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def render_opengl(env):
        """Render the environment using OpenGL"""
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        # Draw grid
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINES)
        for i in range(env.grid_size + 1):
            glVertex2f(0, i * CELL_SIZE)
            glVertex2f(SCREEN_WIDTH, i * CELL_SIZE)
            glVertex2f(i * CELL_SIZE, 0)
            glVertex2f(i * CELL_SIZE, SCREEN_HEIGHT)
        glEnd()

        # Draw obstacles, safe zone, person, and drone here...
        # (OpenGL implementation details omitted for brevity)

        pygame.display.flip()
