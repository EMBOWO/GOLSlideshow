import pygame
import numpy as np
import sys
import time
import math
from scipy import signal
from PIL import Image, ImageDraw, ImageFont

# Configuration
WIDTH, HEIGHT = 1536, 960  # Default slide dimensions

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class GameOfLife:
    def __init__(self, width, height, cell_size, initial_grid=None, wrap=False, slide_offset=(0,0), rule="B3/S23"):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = math.ceil((self.width + slide_offset[1]) / self.cell_size)
        self.rows = math.ceil((self.height + slide_offset[0]) / self.cell_size)
        self.slide_offset = slide_offset
        
        # Initialize grid with the provided grid or as all dead
        self.grid = initial_grid.copy() if initial_grid is not None else np.zeros((self.rows, self.cols), dtype=np.int8)
        if initial_grid is None:
            self.initialize_random()

        # parse wrap
        if wrap:
            self.mode = 'wrap'
        else:
            self.mode = 'constant'
        
        # Parse the rule string into birth and survival sets
        self.births, self.survivals = self.parse_rule(rule)

    def parse_rule(self, rule):
        # Extract birth and survival rules from the rule string "B3/S23"
        birth_rule, survival_rule = rule.split('/')
        
        # Convert birth and survival rules to sets of integers
        births = set(map(int, birth_rule[1:]))
        survivals = set(map(int, survival_rule[1:]))
        
        return births, survivals

    def initialize_random(self):
        # Initialize with random cells (about 25% alive)
        self.grid = np.random.choice([0, 1], size=(self.rows, self.cols), p=[0.75, 0.25])

    def update(self):
        # Create padded grid for boundary conditions
        padded_grid = np.pad(self.grid, pad_width=1, mode=self.mode)
        
        # Calculate neighbor counts using convolution
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        neighbor_counts = signal.convolve2d(padded_grid, kernel, mode='valid')
        
        # Apply the rules dynamically based on the parsed birth and survival sets
        birth = (self.grid == 0) & np.isin(neighbor_counts, list(self.births))
        survive = (self.grid == 1) & np.isin(neighbor_counts, list(self.survivals))
        
        # Update the grid based on the rules
        self.grid = np.where(birth | survive, 1, 0)

    def invert_color(self, surface, mask=None, rel_pos=(0, 0), color=(255, 255, 255)):
        flag = {'sub': pygame.BLEND_RGB_SUB, 'min': pygame.BLEND_RGB_MIN, 'max': pygame.BLEND_RGB_MAX}
        
        if mask is None:
            new_surf = pygame.Surface(surface.get_size())
            new_surf.fill(color)
            new_surf.blit(surface, (0, 0), special_flags=flag['sub'])
            return new_surf
        
        mask = mask.copy()
        surf_size = surface.get_size()
        black_surf = pygame.Surface(surf_size)

        not_inverted_surf = pygame.Surface(surf_size)
        not_inverted_surf.fill((255, 255, 255))
        not_inverted_surf.blit(mask, rel_pos, special_flags=flag['sub'])

        if color != (255, 255, 255):
            colored_surf = pygame.Surface(mask.get_size())
            colored_surf.fill(color)
            mask.blit(colored_surf, (0, 0), special_flags=flag['min'])

        black_surf.blit(mask, rel_pos)
        not_inverted_surf.blit(surface, (0, 0), special_flags=flag['min'])
        black_surf.blit(surface, (0, 0), special_flags=flag['sub'])

        black_surf.blit(not_inverted_surf, (0, 0), special_flags=flag['max'])
        return black_surf

    def draw(self, surface, original_image):
        # Create a surface for the original image to avoid modifying the source
        image_copy = original_image.copy()

        # Create a mask surface to hold the regions that need to be inverted (alive cells)
        mask_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        mask_surface.fill((0, 0, 0))  # Start with a fully transparent mask

        # Iterate over the grid to find alive cells and create a mask
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 1:  # If cell is alive
                    # Calculate the position and size of the cell on the surface
                    x_start = j * self.cell_size - self.slide_offset[1]
                    y_start = i * self.cell_size - self.slide_offset[0]

                    # Draw a white rectangle (fully visible area) on the mask surface for each alive cell
                    mask_surface.fill((255, 255, 255), pygame.Rect(x_start, y_start, self.cell_size, self.cell_size))

        # Now use the invert_color function to invert the colors of the original image
        inverted_image = self.invert_color(image_copy, mask=mask_surface)

        # Blit the modified (inverted) image back onto the surface
        surface.blit(inverted_image, (0, 0))

class Slideshow:
    def __init__(self):
        pygame.init()
        self.width, self.height = WIDTH, HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Alex - Conway's Game of Life Presentation!")
        self.clock = pygame.time.Clock()
        self.slides = []
        self.current_slide = 0
        self.paused = False
        
    def create_title_image(self, title, captions_list, title_size=72, caption_size=36):
        # Create an image with text
        image = Image.new("RGB", (self.width, self.height), WHITE)
        draw = ImageDraw.Draw(image)
        
        # Create fonts
        title_font = ImageFont.truetype("arial.ttf", title_size)
        
        # Calculate text position to center it
        # Using font.getbbox() instead of the deprecated textsize
        title_bbox = title_font.getbbox(title)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        
        title_position = ((self.width - title_width) // 2, self.height * 2 // 5 - title_height // 2)

        # Draw text
        draw.text(title_position, title, BLACK, font=title_font)

        # repeat for captions
        repeated_height = 0
        for i in range(len(captions_list)):
            caption = captions_list[i]
            caption_font = ImageFont.truetype("arial.ttf", caption_size)

            caption_bbox = caption_font.getbbox(caption)
            caption_width = caption_bbox[2] - caption_bbox[0]
            caption_height = caption_bbox[3] - caption_bbox[1]

            if i == 0:
                repeated_height = caption_height

            caption_position = ((self.width - caption_width) // 2, self.height * 2 // 5 + title_height * 3 // 2 + i * repeated_height * 3 // 2)

            draw.text(caption_position, caption, BLACK, font=caption_font)

        # Convert PIL Image to Pygame Surface
        mode = image.mode
        size = image.size
        data = image.tobytes()
        return pygame.image.fromstring(data, size, mode)

    def create_slide_image(self, title, text_list, title_size=120, text_size=48, margin=50):
        # Create an image with text
        image = Image.new("RGB", (self.width, self.height), WHITE)
        draw = ImageDraw.Draw(image)
        
        # Create fonts
        title_font = ImageFont.truetype("arial.ttf", title_size)
        
        # Calculate text position to center it
        # Using font.getbbox() instead of the deprecated textsize
        title_bbox = title_font.getbbox(title)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        
        title_position = (margin, margin)

        # Draw text
        draw.text(title_position, title, BLACK, font=title_font)

        # repeat for text
        repeated_height = 0
        for i in range(len(text_list)):
            text = text_list[i]
            text_font = ImageFont.truetype("arial.ttf", text_size)

            text_bbox = text_font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if i == 0:
                repeated_height = text_height

            text_position = (margin, margin * 3 // 2 + title_height * 3 // 2 + i * repeated_height * 3 // 2)

            draw.text(text_position, text, BLACK, font=text_font)
        
        # Convert PIL Image to Pygame Surface
        mode = image.mode
        size = image.size
        data = image.tobytes()
        return pygame.image.fromstring(data, size, mode)

    def add_slide(self, image, initial_grid=None, wrap=False, cell_size=50, fps=10, paused=False, slide_offset=(0,0), rule="B3/S23"):
        # Add a slide to the slideshow with an optional initial grid for the Game of Life and custom cell size
        if isinstance(image, str):  # If it's a path to an image
            try:
                slide = pygame.image.load(image)
                slide = pygame.transform.scale(slide, (self.width, self.height))
                self.slides.append((slide, cell_size, initial_grid, wrap, fps, paused, slide_offset, rule))  # Add as a tuple
            except pygame.error as e:
                print(f"Could not load image: {e}")
        else:  # If it's already a Pygame surface
            self.slides.append((image, cell_size, initial_grid, wrap, fps, paused, slide_offset, rule))  # Add as a tuple

    def add_slide_pattern(self, image, rle_str, cols=None, rows=None, offset=(0,0), cell_size=50, wrap=False, fps=10, paused=False, slide_offset=(0,0), rule="B3/S23"):
        pattern = self.parse_rle(rle_str, cols, rows, offset, cell_size, slide_offset)
        self.add_slide(image, pattern, wrap, cell_size, fps, paused, slide_offset, rule)

    def next_slide(self):
        # Go to the next slide
        self.current_slide = (self.current_slide + 1) % len(self.slides)

    def previous_slide(self):
        # Go to the previous slide
        self.current_slide = (self.current_slide - 1) % len(self.slides)

    def initialize_slide(self, slide):
        current_slide, cell_size, initial_grid, wrap, fps, paused, slide_offset, rule = slide
        if initial_grid is not None:
            # Initialize the Game of Life with a custom grid for this slide
            self.game = GameOfLife(self.width, self.height, cell_size, initial_grid, wrap, slide_offset, rule)
        else:
            # Initialize the Game of Life with a random grid
            self.game = GameOfLife(self.width, self.height, cell_size, wrap=wrap, slide_offset=slide_offset, rule=rule)
        self.paused = paused

    def run(self):
        running = True
        reset = False
        self.paused = False  # Flag to pause the simulation
        
        self.initialize_slide(self.slides[0]) # initialize first slide

        dragging = -1

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_RIGHT:
                        self.next_slide()
                        self.initialize_slide(self.slides[self.current_slide])
                    elif event.key == pygame.K_LEFT:
                        self.previous_slide()
                        self.initialize_slide(self.slides[self.current_slide])
                    elif event.key == pygame.K_SPACE:
                        self.initialize_slide(self.slides[self.current_slide])
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_l and self.paused:
                        self.game.update()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_x, mouse_y = pygame.mouse.get_pos()

                        # Account for the slide_offset when calculating grid position
                        offset_x, offset_y = self.slides[self.current_slide][6]  # slide_offset
                        grid_x = (mouse_x - offset_x) // self.slides[self.current_slide][1]  # Convert to grid coordinates (column)
                        grid_y = (mouse_y - offset_y) // self.slides[self.current_slide][1]  # Convert to grid coordinates (row)

                        # Toggle the state of the cell at (grid_y, grid_x)
                        if 0 <= grid_x < self.game.cols and 0 <= grid_y < self.game.rows:
                            self.game.grid[grid_y, grid_x] = 1 - self.game.grid[grid_y, grid_x]  # Toggle between 1 (alive) and 0 (dead)
                            dragging = self.game.grid[grid_y, grid_x]  # Start dragging

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        dragging = -1  # Stop dragging

                elif event.type == pygame.MOUSEMOTION:
                    if dragging >= 0:  # Only update cells when dragging
                        mouse_x, mouse_y = pygame.mouse.get_pos()

                        # Account for the slide_offset when calculating grid position
                        offset_x, offset_y = self.slides[self.current_slide][6]  # slide_offset
                        grid_x = (mouse_x - offset_x) // self.slides[self.current_slide][1]  # Convert to grid coordinates (column)
                        grid_y = (mouse_y - offset_y) // self.slides[self.current_slide][1]  # Convert to grid coordinates (row)

                        # Toggle the state of the cell at (grid_y, grid_x)
                        if 0 <= grid_x < self.game.cols and 0 <= grid_y < self.game.rows:
                            self.game.grid[grid_y, grid_x] = dragging  # Toggle between 1 (alive) and 0 (dead)

            image = self.slides[self.current_slide][0]

            # Draw the Game of Life, inverting colors
            self.game.draw(self.screen, image)

            # Update the display
            pygame.display.flip()
            
            if not self.paused:
                # Update the Game of Life
                self.game.update()
            
            # Control the frame rate
            fps = self.slides[self.current_slide][4]
            self.clock.tick(fps)
        
        pygame.quit()
        sys.exit()

    def parse_rle(self, rle_str, cols=None, rows=None, offset=(0,0), cell_size=50, slide_offset=(0,0)):
        """
        Parses an RLE string and returns a pattern in a numpy array.

        Args:
        - rle_str (str): RLE-encoded string representing the pattern.
        - width (int): The width of the grid in pixels.
        - height (int): The height of the grid in pixels.
        - cell_size (int): The size of each cell in pixels.

        Returns:
        - numpy.ndarray: A grid representing the pattern, with 1s for alive cells and 0s for dead cells.
        """

        total_cols = math.ceil((self.width + slide_offset[1]) / cell_size)
        total_rows = math.ceil((self.height + slide_offset[0]) / cell_size)

        if cols is None:
            cols = total_cols
        if rows is None:
            rows = total_rows

        # Initialize a grid of all dead cells
        pattern = np.zeros((total_rows, total_cols), dtype=np.int8)
        
        # Split the RLE string into rows and decode them
        lines = rle_str.strip().splitlines()

        # Initialize variables for tracking current row and column in the grid
        current_row = 0
        current_col = 0

        last_increment = False

        for line in lines:
            line = line.strip()
            
            if line.startswith('#') or not line:
                continue  # Skip comments and empty lines
            
            # RLE decoding
            i = 0
            while i < len(line):
                count = 1  # Default count
                if line[i].isdigit():
                    # Parse the number (run length)
                    count_str = ''
                    while i < len(line) and line[i].isdigit():
                        count_str += line[i]
                        i += 1
                    count = int(count_str)
                
                # The character after the number is 'b' (for dead) or 'o' (for alive) or '$' (for line break)
                character = line[i]
                i += 1

                if character == '!':
                    break

                if character == '$' and last_increment:
                    count -= 1
                last_increment = False

                # Mark the cells in the pattern based on the decoded run
                for _ in range(count):
                    if character == '$':
                        current_row += 1
                        current_col = 0
                    else:
                        if character == 'o' and current_row < rows and current_col < cols:
                            pattern[current_row + offset[0], current_col + offset[1]] = 1  # Alive cell
                        
                        current_col += 1

                        if current_col >= cols:
                            current_col = 0
                            current_row += 1
                            last_increment = 1

                        if current_row >= rows:
                            break

            if current_row >= rows:
                break
        
        return pattern

if __name__ == "__main__":
    slideshow = Slideshow()
    
    # slide 1
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Conway's Game of Life", ["Alex"], title_size=120, caption_size=60),
        """
        2ob2o22b2ob2o$2ob2o22b2ob2o$2bo26bo$2o28b2o$2o28b2o11$2o28b2o$2o28b2o$2bo26bo$2ob2o22b2ob2o$2ob2o22b2ob2o!
        """,
        cell_size=48,
        paused=True)

    # slide 2
    slideshow.add_slide_pattern(
        "slides\\slide2.png",
        """
        2bo4bo2$bo6bo$2b6o!
        """, offset=(8, 20),
        cell_size=48,
        paused=True)

    # slide 3
    slideshow.add_slide_pattern(
        "slides\\slide3.png",
        """
        32o2$32o!
        """, offset=(16, 0),
        cell_size=48,
        paused=True)

    # slide 4.1
    slideshow.add_slide_pattern(
        slideshow.create_title_image("And what does it look like?", [], title_size=128),
        """
        """)

    # slide 4.2
    slideshow.add_slide(
        slideshow.create_title_image("And what does it look like?", [], title_size=128),
        cell_size=4, fps=30)

    # slide 5
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Wow, that's a lot...", ["Now, let's get into the coolness!"], title_size=120, caption_size=60),
        """
        9bo7bo9b$3b2obobob2o3b2obobob2o3b$3obob3o9b3obob3o$o3bobo5bobo5bobo3bo
$4b2o6bobo6b2o4b$b2o9bobo9b2ob$b2ob2o15b2ob2ob$5bo15bo!
        """, offset=(0, 3),
        cell_size=48, wrap=True, slide_offset=(0,24))

    # slide 6
    slideshow.add_slide(
        slideshow.create_title_image("Spaceships", [], title_size=120))

    # slide 7
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Glider",
            ["Type: spaceship",
            "Period: 4",
            "Speed: c/2 diagonal",
            "Frequency class: 1.8",
            "Heat: 4",
            "Notes: The most common spaceship in Life.",
            "we will go over these stats more thoroughly later."], title_size=120, caption_size=40),
        """
        bo$2bo$3o!
        """, offset=(3,3),
        cell_size=48, wrap=True)

    # slide 8
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Lightweight spaceship (LWSS)",
            ["Type: spaceship",
            "Period: 4",
            "Speed: c/2 | 2c/4",
            "Frequency class: 11.2",
            "Heat: 11",
            "Notes: Part of a family of spaceships (XWSS)."], title_size=100, caption_size=40),
        """
        bo2bo$o4b$o3bo$4o!
        """, offset=(1,20),
        cell_size=48, wrap=True)

    # slide 9
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Middleweight spaceship (MWSS)",
            ["Type: spaceship",
            "Period: 4",
            "Speed: c/2 | 2c/4",
            "Frequency class: 13.2",
            "Heat: 15",
            "Notes: Part of a family of spaceships (XWSS)."], title_size=100, caption_size=40),
        """
        3bo2b$bo3bo$o5b$o4bo$5o!
        """, offset=(1,20),
        cell_size=48, wrap=True)

    # slide 10
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Heavyweight spaceship (HWSS)",
            ["Type: spaceship",
            "Period: 4",
            "Speed: c/2 | 2c/4",
            "Frequency class: 15.7",
            "Heat: 19",
            "Notes: Part of a family of spaceships (XWSS)."], title_size=100, caption_size=40),
        """
        3b2o2b$bo4bo$o6b$o5bo$6o!
        """, offset=(1,20),
        cell_size=48, wrap=True)

    # slide 11
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Schick engine",
            ["Type: spaceship",
            "Period: 12",
            "Speed: c/2 | 6c/12",
            "Frequency class: 48",
            "Heat: 43.2",
            "Notes: Two XWSS pulling a tagalong."], title_size=120, caption_size=40),
        """
        bo2bo$o$o3bo$4o9b2o$6b3o5b2o$6b2ob2o6b3o$6b3o5b2o$4o9b2o$o3bo$o$bo2bo!
        """, offset=(1,20),
        cell_size=24, wrap=True)

    # slide 12
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Loafer",
            ["Type: spaceship",
            "Period: 7",
            "Speed: c/7",
            "Frequency class: 49",
            "Heat: 14.6",
            "Notes: Rarest natural spaceship."], title_size=120, caption_size=40),
        """
        b2o2bob2o$o2bo2b2o$bobo$2bo$8bo$6b3o$5bo$6bo$7b2o!
        """, offset=(4,20),
        cell_size=24, wrap=True)

    # slide 13
    slideshow.add_slide(
        slideshow.create_title_image("Some more spaceships I like", [], title_size=120))

    # slide 14
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Weekender",
            ["Type: spaceship",
            "Period: 7",
            "Speed: 2c/7",
            "Heat: 45.1",
            "Notes: Interesting speed. It looks funny."], title_size=120, caption_size=40),
        """
        bo12bob$bo12bob$obo10bobo$bo12bob$bo12bob$2bo3b4o3bo2b$6b4o6b$2b4o4b4o
2b2$4bo6bo4b$5b2o2b2o!
        """, offset=(8,24),
        cell_size=24, wrap=True)

    # slide 15
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Swan",
            ["Type: spaceship",
            "Period: 4",
            "Speed: c/4 diagonal",
            "Heat: 51.0",
            "Notes: Cool looking."], title_size=120, caption_size=40),
        """
        bo10b2o10b$5o6b2o11b$o2b2o8bo7b2ob$2b2obo5b2o6b3obo$11b2o3bob2o4b$5bob
o6b2o8b$10b3obo4bo4b$7b3o3bo4bo5b$8bo7bo7b$8bo6bo8b2$11bo!
        """, offset=(8,24),
        cell_size=24, wrap=True)

    # slide 16
    slideshow.add_slide_pattern(
        slideshow.create_title_image("232P7H3V0",
            ["Type: spaceship",
            "Period: 7",
            "Speed: 3c/7",
            "Heat: 214.3",
            "Notes: Some spaceships are unnamed. Some are big."], title_size=120, caption_size=40),
        """
        17b3o9b3o$16bo3bo7bo3bo$15bobo3bo5bo3bobo$15bo3b2o7b2o3bo$15b3o3bo5bo
3b3o$14bo3b2ob3ob3ob2o3bo$14b2o2bo3b2ob2o3bo2b2o$13b3o3b5ob5o3b3o$21bo
5bo$18bo11bo$13bo4bo11bo4bo$13bo4b4o5b4o4bo$17bo4bo3bo4bo$16b2ob3o5b3o
b2o$16b2obo3bobo3bob2o$12b3o7b2ob2o7b3o$11bo3b2o4bobobobo4b2o3bo$10bo
3bo19bo3bo$10bo9b3o3b3o9bo$14bo6b2o3b2o6bo$9bo12bo3bo12bo$9bo2b2o5bo9b
o5b2o2bo$10b2o8bo7bo8b2o$8bo12bo5bo12bo$7b3o29b3o$6b2o2bo27bo2b2o$9b2o
27b2o$9bo29bo2$6bo35bo$7b2o31b2o$5bo2bo31bo2bo$4bo39bo$3b2o39b2o$2b4o
37b4o$bo45bo$b3o41b3o$o47bo$2b2o41b2o$4bo39bo$2b2o41b2o$3bo41bo$2bo43b
o$2bo43bo!
        """, offset=(6,40),
        cell_size=12, wrap=True, fps=10, slide_offset=(0,6))

    # slide 17
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Copperhead",
            ["Type: spaceship",
            "Period: 10",
            "Speed: c/10",
            "Heat: 27.8",
            "Notes: Surprisingly small and high period."], title_size=120, caption_size=40),
        """
        b2o2b2o$3b2o$3b2o$obo2bobo$o6bo2$o6bo$b2o2b2o$2b4o2$3b2o$3b2o!
        """, offset=(6,28),
        cell_size=24, wrap=True, fps=10)

    # slide 18
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Sir Robin",
            ["Type: spaceship",
            "Period: 6",
            "Speed: (2,1)c/6",
            "Heat: 271",
            "Notes: First ever discovered knightship (!!)"], title_size=120, caption_size=40),
        """
        4b2o$4bo2bo$4bo3bo$6b3o$2b2o6b4o$2bob2o4b4o$bo4bo6b3o$2b4o4b2o3bo$o9b
2o$bo3bo$6b3o2b2o2bo$2b2o7bo4bo$13bob2o$10b2o6bo$11b2ob3obo$10b2o3bo2b
o$10bobo2b2o$10bo2bobobo$10b3o6bo$11bobobo3bo$14b2obobo$11bo6b3o2$11bo
9bo$11bo3bo6bo$12bo5b5o$12b3o$16b2o$13b3o2bo$11bob3obo$10bo3bo2bo$11bo
4b2ob3o$13b4obo4b2o$13bob4o4b2o$19bo$20bo2b2o$20b2o$21b5o$25b2o$19b3o
6bo$20bobo3bobo$19bo3bo3bo$19bo3b2o$18bo6bob3o$19b2o3bo3b2o$20b4o2bo2b
o$22b2o3bo$21bo$21b2obo$20bo$19b5o$19bo4bo$18b3ob3o$18bob5o$18bo$20bo$
16bo4b4o$20b4ob2o$17b3o4bo$24bobo$28bo$24bo2b2o$25b3o$22b2o$21b3o5bo$
24b2o2bobo$21bo2b3obobo$22b2obo2bo$24bobo2b2o$26b2o$22b3o4bo$22b3o4bo$
23b2o3b3o$24b2ob2o$25b2o$25bo2$24b2o$26bo!
        """, offset=(40,100),
        cell_size=6, wrap=True, fps=20)

    # slide 19
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Walrus",
            ["Type: spaceship",
            "Period: 8",
            "Speed: c/8 diagonal",
            "Heat: 72.5",
            "Notes: First elementary c/8 spaceship"], title_size=120, caption_size=40),
        """
        4b6o$6b2o$3bo5b2o$b8obo$4o3bo$o9b3o$3b2o3b2ob2o2b2o$2b3o2b5ob3o2bo$bob
2obobo2b3o4bo$2obo3bo4b3o4bo$7bo$2bo$3b2o$8b3o$8bo2$9bo$6b2ob2o!
        """, offset=(20,50),
        cell_size=12, wrap=True)

    # slide 19.5
    slideshow.add_slide_pattern(
        slideshow.create_title_image("So, what's it mean to be natural?",
            ["Soup search is a process where random squares (soups) are generated,",
            "Then, patterns are picked out of the soup.",
            "A pattern is natural if it has occurred in a soup before.",
            "Frequency class x of an object O is defined such that the most",
            "common object is 2^x times more common than O.",
            "Pictured: one of the three times loafer has naturally occurred."], title_size=100, caption_size=40),
        """
        bbooooobbobbbboo$
bboboooobobbboob$
bbbbbbbbboboooob$
obbbooobobbooboo$
bobboboboobobooo$
obbbbobooobboooo$
boobbbobbobboobb$
obbobooooboboboo$
booobbbbboobbbob$
obobobobooboooob$
oboboooooobbboob$
obboooboooboobob$
boboobbobbbooobo$
bobbbooobboobbbb$
bbbobbobboobobbb$
oboooobbbbbobbob!
        """, offset=(28,50),
        cell_size=12, paused=True, wrap=True)

    # slide 20
    slideshow.add_slide(
        slideshow.create_title_image("Still lifes and oscillators", [], title_size=120))

    # slide 21.1
    slideshow.add_slide_pattern(
        slideshow.create_slide_image("Still lifes and oscillators",
            ["Still lifes are the most common objects, along with oscillators",
            "Still lifes are oscillators of period 1",
            "Strict still lifes must not have stable strict subsets"]),
        """
        2o8b2o4b2o4bo4b2o3bo$2o2b3o2bo2bo2bo2bo2bobo2bobo2bobo$10b2o3bobo3b2o
3b2o4bo$16bo$b2o5bo$o2bo3bobo3b2o3b2o6bo3b2o$o2bo2bobo3bo5b2o5bobobo2b
o$b2o3b2o7bo4b2o2bobo3bo2bo$13b2o5b2o3bo5b2o$9bo$2o6bobo$obo4bobo4b2o
$2bo3bobo3bo2bo6b3o3b3o$2b2o3bo4b2o$20bo4bobo4bo$20bo4bobo4bo$20bo4bo
bo4bo$22b3o3b3o2$22b3o3b3o$20bo4bobo4bo$20bo4bobo4bo$20bo4bobo4bo2$22b
3o3b3o!
        """, offset=(1,6),
        cell_size=36)

    # slide 21.2
    slideshow.add_slide_pattern(
        slideshow.create_slide_image("",
            []),
        """
        2o8b2o4b2o4bo4b2o3bo$2o2b3o2bo2bo2bo2bo2bobo2bobo2bobo$10b2o3bobo3b2o
3b2o4bo$16bo$b2o5bo$o2bo3bobo3b2o3b2o6bo3b2o$o2bo2bobo3bo5b2o5bobobo2b
o$b2o3b2o7bo4b2o2bobo3bo2bo$13b2o5b2o3bo5b2o$9bo$2o6bobo$obo4bobo4b2o
$2bo3bobo3bo2bo6b3o3b3o$2b2o3bo4b2o$20bo4bobo4bo$20bo4bobo4bo$20bo4bo
bo4bo$22b3o3b3o2$22b3o3b3o$20bo4bobo4bo$20bo4bobo4bo$20bo4bobo4bo2$22b
3o3b3o!
        """, offset=(1,6),
        cell_size=36)

    # slide 22
    slideshow.add_slide(
        slideshow.create_title_image("More oscillators", [], title_size=120))

    # slide 23
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Candelabra",
            ["Type: oscillator",
            "Period: 3",
            "Volatility: 0.19",
            "Heat: 4"], title_size=120, caption_size=40),
        """
        4b2o4b2o4b$bo2bo6bo2bob$obobo6bobobo$bo2bob4obo2bob$4bobo2bobo4b$5bo4b
o!
        """, offset=(6,24),
        cell_size=24)

    # slide 24
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Buckaroo",
            ["Type: oscillator",
            "Period: 30",
            "Heat: 16.7",
            "Notes: This is a reflector; it can change the direction of an incoming glider."], title_size=120, caption_size=40),
        """
        9bo$7bobo$6bobo$2o3bo2bo$2o4bobo$7bobo9b2o$9bo9bobo$21bo$21b2o!
        """, offset=(6,24),
        cell_size=24)

    # slide 25
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Turning toads",
            ["Type: oscillator",
            "Period: 4",
            "Notes: Repeatedly flips direction (could be described as rotation)."], title_size=120, caption_size=40),
        """
        15bo6bo14b$14b2o5b2o6b2o6b$6b3obobob2obobob2obobo10b$2b2obo6bobo4bobo
4bobo2bob2o2b$o2bobo3bo18b4obo2bo$2obobo27bob2o$3bo29bo3b$3b2o27b2o!
        """, offset=(6,16),
        cell_size=24)

    # slide 26
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Phoenices (plural of phoenix)",
            ["Type: oscillator",
            "Has volatility 1"], title_size=100, caption_size=40),
        """
        4bo16bo19bo15bo19bo12bo17bo17bo12bo21bo7bo17bo7bo$2bobo16bobo17bobo13b
obo17bobo8bobo15bobo15bobo12bobo19bobo5bobo15bobo5bobo$6bo12bo19bo5bo
9bo17bobo16bo11bo5bo11bo5bo6bobo5bo15bo5bobo17bo5bobo$2o22b2o19bobo12b
2o11bo6b2o4b2o14bobo15bobo12bo7bobo19bo6b2o17bo6b2o$6b2o10b2o18b2o14b
2o15bo20b2o16b2o6bo9b2o3bo24b2o24b2o$bo23bo22b2o11bo19bo5bo12b2o14bob
o27b2o24bo25bo$3bobo14bo19bobo13bobo4bo6b2o22bo14bo15bobo4b2o26bo25bo
$3bo22b2o14bo6bo8bo4bobo16b2o4b2o10bo13b2o9bo20bo23b2o24b2o$20b2o22bo
15bo11bo21b2o12b2o11bobo10bobo7bo15b2o24b2o$26bo23b2o14b2o14bo5bo9b2o
15bo5bo14bo5bobo25bo26bo$22bobo19b2o14b2o10b2o6bo12bo11bobo9bobo18bob
o21bo22bo$24bo25bo15bo11bobo5b2o11bo5bo11bo22bo27b2o28b2o$46bobo13bob
o9bobo15b2o7bobo58b2o20b2o6bo$48bo15bo11bo10bo13bo66bo21bobo5bo$89bob
o72bobo19bobo5bobo$89bo76bo21bo7bo!
        """, offset=(15,6),
        cell_size=7, fps=5)

    # slide 27
    slideshow.add_slide(
        slideshow.create_title_image("More patterns", [], title_size=120))

    # slide 28
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Agar",
            ["Patterns that can tile space and are stable in both space and time",
            "Pictured: chicken wire"], title_size=120, caption_size=40),
        """
        2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o
4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b
2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o
2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b
2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o
2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b
2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o
2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b
4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o
4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b
3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o
3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b
2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o
2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b
5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o
5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b
3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o
3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b
2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o
2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b
3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o
$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b
4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$
2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o2b2o2b4o3b2o5b3o2b3o4b
4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b2o2b2o4b3o2b5o3b2o$2o
2b2o2b4o3b2o5b3o2b3o4b4o2b2o2b4o3b2o5b3o2bo$2b2o2b2o4b3o2b5o3b2o3b4o4b
2o2b2o4b3o2b5o3b2o!
        """,
        cell_size=24, wrap=True)

    # slide 29
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Negative spaceship",
            ["Spaceships that travel through agar instead of space"], title_size=120, caption_size=40),
        """
        53o2$27o2b3o2b19o$28b2o3b2o$22o2b3o3b2o3b4o2b12o$23b2o15b2o$22o2b3o3b
2o5b2o3b2o2b7o$28b2o3bob2o8b2o$22o2b3o2b3o3b2o5b2o3b6o$23b2o12b2ob2o$
22o3b5o10b2o5b6o$26bo15b2ob2o$22o5bo9bo7b2o2b4o$23bobo6bo3bo10b2o$22o
3bobob5o6bo7b5o$28b3o3bo6bo$22o3bobob5o6bo7b5o$23bobo6bo3bo10b2o$22o5b
o9bo7b2o2b4o$26bo15b2ob2o$22o3b5o10b2o5b6o$23b2o12b2ob2o$22o2b3o2b3o3b
2o5b2o3b6o$28b2o3bob2o8b2o$22o2b3o3b2o5b2o3b2o2b7o$23b2o15b2o$22o2b3o
3b2o3b4o2b12o$28b2o3b2o$27o2b3o2b19o2$53o2$53o!
        """,
        cell_size=29, wrap=True)

    # slide 30
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Negative spaceship",
            ["This one's against the grain!"], title_size=120, caption_size=40),
        """
        obobobobobobobobobobobobobobobobobobobobobobobobobo$obobobobobobobobo
bobobobobobobobobobobobobobobobobo$obobobobobobobobobobobobobobobobob
obobobobobobobobo$obobobobobobobobobobobobobobobobobobobobobobobobobo
$obobobobobobobobobobobobobobobobobobobobobobobobobo$obobobobobobobob
obobobobobobobobobobo2b2obobobobobo$obobobobobobobobobobobobobobobobo
bobo4b2obobobobo$obobobobobobobobobobobobobo2bo2bobo7bobobobobo$obobo
bobobobobobobobobobobo4b2obo8b2obobobo$obobobobobobobobobobobobob2o5b
2obo9bobobo$obobobobobobobobobobobobob2o6bobo2bobo3b2obobo$obobobobob
obobobobobobobobo2bo5b2obo2bobo2bobobo$obobobobobobobobobobo2b2obo2bo
5bo2bo5bobobobo$obobobobobobobobobo5bobobo7bobo3b3obobobo$obobobobobo
bobobobo6b2obo2bobo3bo2b2obo2bobobo$obobobobobobobo2b2o8b2o3b3o2b2o3b
o3bobobo$obobobobobobobo5bo7b3o5b2o4bo4b2obo$obobobobobobo7b2o11bobo5b
o5bo2bo$obobobobobobo7b2o11bobo5bo5bo2bo$obobobobobobobo5bo7b3o5b2o4b
o4b2obo$obobobobobobobo2b2o8b2o3b3o2b2o3bo3bobobo$obobobobobobobobobo
6b2obo2bobo3bo2b2obo2bobobo$obobobobobobobobobo5bobobo7bobo3b3obobobo
$obobobobobobobobobobo2b2obo2bo5bo2bo5bobobobo$obobobobobobobobobobob
obobo2bo5b2obo2bobo2bobobo$obobobobobobobobobobobobob2o6bobo2bobo3b2o
bobo$obobobobobobobobobobobobob2o5b2obo9bobobo$obobobobobobobobobobob
obobo4b2obo8b2obobobo$obobobobobobobobobobobobobo2bo2bobo7bobobobobo$
obobobobobobobobobobobobobobobobobobo4b2obobobobo$obobobobobobobobobo
bobobobobobobobobo2b2obobobobobo$obobobobobobobobobobobobobobobobobob
obobobobobobobo$obobobobobobobobobobobobobobobobobobobobobobobobobo$o
bobobobobobobobobobobobobobobobobobobobobobobobobo$obobobobobobobobob
obobobobobobobobobobobobobobobobo$obobobobobobobobobobobobobobobobobo
bobobobobobobobo$obobobobobobobobobobobobobobobobobobobobobobobobobo$
obobobobobobobobobobobobobobobobobobobobobobobobobo!
        """,
        cell_size=30, wrap=True)

    # slide 31
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Greyship",
            ["A spaceship that contains agar"], title_size=120, caption_size=40),
        """
        68bo$65b4o$65b3o$63bo3bo$63b3o10b2o$61bo5bo7b2ob2o$61b6o6bob5o$59bo10b
obob2ob2o$59b12obob2o$57bo16b2o$57b18o9b4o$55bo19bo7bo4bo$55b20o7bo$
53bo23bobobo6bo$53b27o$51bo42b2o$51b32o10b4o$49bo34bo7b5o$49b35o7bo$
47bo38bob2o$47b40ob2o$45bo44bo11bo2bo$45b46o10bo$7bo35bo57bo3bo$4b4o
35b50o8b4o$4b2o35bo53bob3o$2bo38b55obobo$2b4o33bo59bo$bo37b61o10b2o$3o
2bo31bo71b2ob2o$b3o33b64o6bob5o$2bo32bo68bobob2ob2o$3b3obo27b70obob2o$
3bo2b2o25bo74b2o$4b3o26b76o9b4o$6bo24bo77bo7bo4bo$4bo26b78o7bo$4bobo
22bo81bobobo6bo$3bo25b85o$4bobo20bo100b2o$4bo22b90o10b4o$6bo18bo92bo7b
5o$4b3o18b93o7bo$3bo2b2o15bo96bob2o$3b3obo15b98ob2o$2bo18bo102bo11bo2b
o$b3o17b104o10bo$3o2bo13bo115bo3bo$bo17b108o8b4o$2b4o11bo111bob3o$2bo
14b113obobo$4b2o9bo117bo$4b4obo5b119o$7b2obo2bo$9b2o2b122o$10b2o126bo$
11b128o2$11b128o$10b2o126bo$9b2o2b122o$7b2obo2bo$4b4obo5b119o$4b2o9bo
117bo$2bo14b113obobo$2b4o11bo111bob3o$bo17b108o8b4o$3o2bo13bo115bo3bo$
b3o17b104o10bo$2bo18bo102bo11bo2bo$3b3obo15b98ob2o$3bo2b2o15bo96bob2o$
4b3o18b93o7bo$6bo18bo92bo7b5o$4bo22b90o10b4o$4bobo20bo100b2o$3bo25b85o
$4bobo22bo81bobobo6bo$4bo26b78o7bo$6bo24bo77bo7bo4bo$4b3o26b76o9b4o$3b
o2b2o25bo74b2o$3b3obo27b70obob2o$2bo32bo68bobob2ob2o$b3o33b64o6bob5o$
3o2bo31bo71b2ob2o$bo37b61o10b2o$2b4o33bo59bo$2bo38b55obobo$4b2o35bo53b
ob3o$4b4o35b50o8b4o$7bo35bo57bo3bo$45b46o10bo$45bo44bo11bo2bo$47b40ob
2o$47bo38bob2o$49b35o7bo$49bo34bo7b5o$51b32o10b4o$51bo42b2o$53b27o$53b
o23bobobo6bo$55b20o7bo$55bo19bo7bo4bo$57b18o9b4o$57bo16b2o$59b12obob2o
$59bo10bobob2ob2o$61b6o6bob5o$61bo5bo7b2ob2o$63b3o10b2o$63bo3bo$65b3o$
65b4o$68bo!
        """, offset=(12,6),
        cell_size=6, wrap=True, fps=20)

    # slide 32
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Puffers",
            ["Spaceships that leave debris behind them",
            "Pictured: space rake (rakes leave spaceships behind)",
            "Puffers are an example of infinite growth."], title_size=120, caption_size=40),
        """
        11b2o5b4o$9b2ob2o3bo3bo$9b4o8bo$10b2o5bo2bob2$8bo13b$7b2o8b2o3b$6bo9bo
2bo2b$7b5o4bo2bo2b$8b4o3b2ob2o2b$11bo4b2o4b4$18b4o$o2bo13bo3bo$4bo16bo
$o3bo12bo2bob$b4o!
        """, offset=(100,100),
        cell_size=6, wrap=True, fps=20)

    # slide 33
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Infinite growth",
            ["Comes in linear, quadratic, log, etc. varieties",
            "Here is the smallest starting configuration (10 cells)"], title_size=120, caption_size=40),
        """
        6bob$4bob2o$4bobob$4bo3b$2bo5b$obo!
        """, offset=(130,200),
        cell_size=6, wrap=True, paused=True, fps=60)

    # slide 34
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Methuselahs",
            ["Methuselahs are patterns that take a long time to stabilize (\"go boring\") for their size.",
            "Here's a small one (R-pentomino, 1103 generations to stabilize)"], title_size=120, caption_size=40),
        """
        b2o$2ob$bo!
        """, offset=(130,200),
        cell_size=6, wrap=True, paused=True, fps=60)

    # slide 35
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Guns",
            ["Guns produce other patterns",
            "Here's the most famous, the one and only Gosper glider gun",
            "Also exhibit infinite growth"], title_size=120, caption_size=40),
        """
        24bo11b$22bobo11b$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o14b$2o8b
o3bob2o4bobo11b$10bo5bo7bo11b$11bo3bo20b$12b2o!
        """, offset=(12, 20),
        cell_size=10, fps=20)

    # slide 36
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Gardens of Eden",
            ["To understand this one, we must understand the Garden of Eden theorem."], title_size=120, caption_size=40),
        """
        33o$2obob3ob3ob2obobobobobobobobobo$obob3ob3ob4ob3obobobobobobob$5ob3o
b3ob4ob14o$obob2ob3ob3obob3obobobobobobob$4ob3ob3ob5ob2obobobobobobo$b
2ob3ob3ob3obobob13o$2ob2ob3ob3ob2ob4obobobobobobo$18ob14o!
        """, offset=(4, 16),
        cell_size=24, paused=True)

    # slide 37
    slideshow.add_slide(
        slideshow.create_title_image("Interesting concepts", [], title_size=120))

    # slide 38
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Speed of light",
            ["Finally, we come around to the speed of light, c.",
            "Cells look in a 3x3 neighborhood for their state in the next tick, so c = 1 cell/tick.",
            "Spaceships can travel at c/2 orthogonally or diagonally.",
            "However, other patterns travel closer to c (pictured: a wick, \"ants\")"], title_size=120, caption_size=40),
        """
        2o3b2o3b2o3b2o3b2o3b2o3b2o3b2o3b2o2b$2b2o3b2o3b2o3b2o3b2o3b2o3b2o3b2o
3b2o$2b2o3b2o3b2o3b2o3b2o3b2o3b2o3b2o3b2o$2o3b2o3b2o3b2o3b2o3b2o3b2o3b
2o3b2o!
        """, offset=(4, 0),
        cell_size=35, slide_offset=(0,5), wrap=True)

    # slide 39
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Here is a fuse",
            ["The naming is intuitive, and it conveys information at 2c/3."], title_size=120, caption_size=40),
        """
        41bo$6bo32b2o$4bobo33b2o$5b2o2$34bo$33bo$33b3o2$25bo$23b3o$22bo$22b2o
6$26bo4b2o$25bobobo2bo$26b2ob3o2$26b6o$25bo6bo$25bo2b5o$23bobobo7bo$
23b2o2bo2b6o$27bobo$26b2obo2b6o$29bobo6bo$bo27bobo2b5o$b2o27b2obo7bo$o
bo30bo2b6o$33bobo$32b2obo2b6o$35bobo6bo$5bo29bobo2b5o$5b2o29b2obo7bo$
4bobo32bo2b6o$39bobo$38b2obo2b6o$41bobo6bo$41bobo2b5o$42b2obo7bo$45bo
2b6o$45bobo$44b2obo2b6o$47bobo6bo$47bobo2b5o$48b2obo7bo$51bo2b6o$51bob
o$50b2obo2b6o$53bobo6bo$53bobo2b5o$54b2obo7bo$57bo2b6o$57bobo$56b2obo
2b6o$59bobo6bo$59bobo2b3obo$60b2obo3bo$63bo2bo$63bobo$62b2obobo$66b2o!
        """, offset=(6, 30),
        cell_size=12)

    # slide 40.1
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Glider synthesis",
            ["For some reason, people like to make patterns out of gliders colliding.",
            "The fewer, the better.",
            "Pictured: 8-glider loafer synthesis"], title_size=120, caption_size=40),
        """
        33bo$31b2o$32b2o$9bo$bo8bo$2bo5b3o$3o3$5bo$6bo$4b3o$24bobo$25b2o$25bo
2$27bobo$27b2o$28bo$31b3o$31bo$32bo7$5b2o$6b2o$5bo!
        """, offset=(2, 12),
        cell_size=24, paused=True)

    # slide 40.2
    slideshow.add_slide_pattern(
        slideshow.create_title_image("",
            [], title_size=120, caption_size=40),
        """
        33bo$31b2o$32b2o$9bo$bo8bo$2bo5b3o$3o3$5bo$6bo$4b3o$24bobo$25b2o$25bo
2$27bobo$27b2o$28bo$31b3o$31bo$32bo7$5b2o$6b2o$5bo!
        """, offset=(2, 12),
        cell_size=24)

    # slide 41
    slideshow.add_slide(
        slideshow.create_title_image("Constructions", [], title_size=120))

    # slide 42
    slideshow.add_slide_pattern(
        "slides\\slide42.png",
        """
        
        """)

    # slide 43
    slideshow.add_slide_pattern(
        "slides\\slide43.png",
        """
        
        """)

    # slide 44
    slideshow.add_slide_pattern(
        "slides\\slide44.png",
        """
        
        """)

    # slide 45
    slideshow.add_slide_pattern(
        "slides\\slide45.png",
        """
        
        """)

    # slide 46
    slideshow.add_slide_pattern(
        "slides\\slide46.png",
        """
        
        """)

    # slide 47
    slideshow.add_slide(
        slideshow.create_title_image("Other rules", ["Which brings us to other rules."], title_size=120),
        rule="B36/S23")

    # slide 48.1
    slideshow.add_slide_pattern(
        slideshow.create_title_image("HighLife",
            ["Standard Life has births with 3 neighbors and survivals with 2 or 3 neighbors.",
            "This can be encoded as the rulestring B3/S23 (birth 3/survival 2, 3).",
            "HighLife, another rule, has rulestring B36/S23.",
            "It functions similarly to Life, but has several unique features.",
            "One of these is a replicator:"], title_size=120, caption_size=40),
        """
        2b3o$bo2bo$o3bo$o2bob$3o!
        """, offset=(36, 64),
        cell_size=12, paused=True, rule="B36/S23")

    # slide 48.2
    slideshow.add_slide_pattern(
        slideshow.create_title_image("",
            [], title_size=120, caption_size=40),
        """
        2b3o$bo2bo$o3bo$o2bob$3o!
        """, offset=(36, 64),
        cell_size=12, rule="B36/S23")

    # slide 49
    slideshow.add_slide_pattern(
        slideshow.create_title_image("HighLife",
            ["Funnily enough, the only way we have found",
            "to make a replicator in Life is by simulating",
            "other rules using the 0E0P metacell."], title_size=120, caption_size=40),
        """
        2b3o$bo2bo$o3bo$o2bob$3o!
        """, offset=(36, 64),
        cell_size=12, rule="B36/S23")

    # slide 50
    slideshow.add_slide(
        slideshow.create_title_image("Gems", ["B3457/S4568"], title_size=120),
        rule="B3457/S4568")

    # slide 51.1
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Gems",
            ["B3457/S4568",
            "This one has a single spaceship.",
            "Guess its speed!"], title_size=120, caption_size=40),
        """
        5b2o$3bo4bo$3b2o2b2o$b3o4b3o$b3o4b3o$2ob6ob2o$3ob4ob3o$ob3o2b3obo$2ob
6ob2o$obobo2bobobo$2b2ob2ob2o$2b8o$4b4o$4bo2bo!
        """, offset=(3, 3),
        cell_size=48, fps=120, paused=True, rule="B3457/S4568")

    # slide 51.2
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Gems",
            ["B3457/S4568",
            "This one has a single spaceship.",
            "Pictured: c/5648 spaceship"], title_size=120, caption_size=40),
        """
        5b2o$3bo4bo$3b2o2b2o$b3o4b3o$b3o4b3o$2ob6ob2o$3ob4ob3o$ob3o2b3obo$2ob
6ob2o$obobo2bobobo$2b2ob2ob2o$2b8o$4b4o$4bo2bo!
        """, offset=(3, 3),
        cell_size=48, fps=120, rule="B3457/S4568")

    # slide 52
    slideshow.add_slide(
        slideshow.create_title_image("Replicator", ["B1357/S1357"], title_size=120),
        rule="B1357/S1357")

    # slide 53
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Replicator",
            ["B1357/S1357",
            "Here, every pattern is a replicator."], title_size=120, caption_size=40),
        """
        2bo4bo2$bo6bo$2b6o!
        """, offset=(5, 5),
        cell_size=12, paused=True, fps=2, rule="B1357/S1357")

    # slide 54
    slideshow.add_slide(
        slideshow.create_title_image("Maze", ["B3/S12345"], title_size=120),
        rule="B3/S12345")

    # slide 55
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Maze",
            ["B3/S12345",
            "An explosive rule.",
            "Maze-like patterns form well."], title_size=120, caption_size=40),
        """
        2bo4bo2$bo6bo$2b6o!
        """, offset=(5, 5),
        cell_size=12, paused=True, fps=20, rule="B3/S12345")

    # slide 56
    slideshow.add_slide(
        slideshow.create_title_image("Wow", ["That was a lot."], title_size=120),
        rule="B3/S13")

    # slide 57
    slideshow.add_slide(
        slideshow.create_title_image("To conclude", ["I hope you learned something from our tromp through Life",
            "and cellular automata.",
            "As an intellectual exercise,",
            "or a demonstration of computing,",
            "or just cool pictures."], title_size=120),
        rule="B3678/S135678")

    # slide 58
    slideshow.add_slide_pattern(
        slideshow.create_title_image("Thank you.", [], title_size=120),
        """
        3bo7bo$3bobo5bobo$bo5bobo$7bo6b2o$2o$14bo$2bo$12b2o$2b2o$12bo$4bo$10b
2o$4b2o$10bo$6bobo$8bo!
        """, offset=(22,24),
        cell_size=24)

    slideshow.run()