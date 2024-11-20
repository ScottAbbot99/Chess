# definitions
HEIGHT, WIDTH = 512, 512 # board image size
NUM_ROWS = NUM_COLS = 8 # board is 8x8
SQUARE_SIZE = WIDTH / NUM_ROWS # size of each square is 512/8 x 512/8
MAX_FPS = 15 
IMAGES = {} # dictionary to store images

# move log dimensions
MOVE_LOG_HEIGHT = HEIGHT
MOVE_LOG_WIDTH = 250