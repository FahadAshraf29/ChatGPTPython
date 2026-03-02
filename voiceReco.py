import speech_recognition as sr
import pygame
import math
import numpy as np

def get_microphone_level():
    with sr.Microphone() as source:
        r = sr.Recognizer()
        try:
            audio = r.listen(source, timeout=1, phrase_time_limit=5)
            numpydata = np.frombuffer(audio.frame_data, dtype=np.int16)
            return numpydata.ptp() / 2**15
        except (sr.WaitTimeoutError, Exception):
            return 0


def show_animation():
    # Initialize pygame
    pygame.init()

    # Screen dimensions
    WIDTH, HEIGHT = 800, 100

    # Creating a borderless window at the bottom of the screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
    pygame.display.set_caption("Voice Bar Animation")
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME | pygame.SRCALPHA)

    # Colors
    TRANSPARENT = (0, 0, 0, 0)
    BLUE = (0, 0, 255)

    clock = pygame.time.Clock()
    running = True
    angle = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Getting the microphone level
        mic_level = get_microphone_level()

        # Rendering the animation
        screen.fill(TRANSPARENT)

        # Drawing dynamic lines based on audio input
        for i in range(30):
            offset = 50 * mic_level * math.sin(angle + i/1.5)
            pygame.draw.line(screen, BLUE, (i*26, HEIGHT//2), (i*26, HEIGHT//2 + offset), 2)

        angle += 0.2

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    show_animation()
