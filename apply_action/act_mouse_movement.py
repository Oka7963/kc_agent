import win32api
from humancursor.system_cursor import SystemCursor
import time
import random
import math
from logger import setup_logger

logger = setup_logger("mouse_movement")

cursor = SystemCursor()  # Initializing SystemCursor object

def get_current_mouse_position() -> tuple[int, int]:
    """Get current mouse position."""
    import win32gui
    return win32gui.GetCursorPos()

def move_mouse_to_rect(rect: tuple[int, int, int, int],
                        angle_variation_degrees: float = 20.0,
                        overshoot_percentage: float = 5.0) -> None:
    """Move mouse to a position within a rectangle with random offset and overshoot.
    Apply random offset to the center point within 10% of the rectangle dimensions.
    Include overshoot point before final movement for more natural motion.

    Args:
        rect: A tuple of (x, y, width, height) coordinates
        angle_variation_degrees: Maximum angle variation in degrees (default: 20.0)
        overshoot_percentage: Percentage of distance to overshoot (default: 5.0)
    """
    x, y, width, height = rect

    # Calculate center point
    center_x = x + width // 2
    center_y = y + height // 2

    # Calculate random offset (±10% of dimensions)
    offset_x = int((random.random() - 0.5) * width * 0.2)  # ±10% of width
    offset_y = int((random.random() - 0.5) * height * 0.2)  # ±10% of height

    # Calculate target coordinates with random offset
    target_x = center_x + offset_x
    target_y = center_y + offset_y

    # Get current mouse position
    current_x, current_y = get_current_mouse_position()

    # Calculate overshoot point (5% beyond target with random angle variation)
    dx = target_x - current_x
    dy = target_y - current_y

    # Calculate distance and base angle
    distance = (dx**2 + dy**2)**0.5

    # If distance is too small, use direct positioning to avoid humancursor errors
    if distance < 5:  # Minimum 5 pixels to avoid library issues
        win32api.SetCursorPos((target_x, target_y))
        logger.debug(f"Direct positioning due to short distance: ({current_x}, {current_y}) -> ({target_x}, {target_y})")
        return

    base_angle = math.atan2(dy, dx)

    # Add random angle variation within specified degrees (converted to radians)
    angle_variation = random.uniform(-angle_variation_degrees, angle_variation_degrees) * math.pi / 180
    final_angle = base_angle + angle_variation

    # Calculate overshoot distance (specified percentage of original distance)
    overshoot_distance = distance * (overshoot_percentage / 100.0)

    # Calculate overshoot point with angle variation
    overshoot_x = target_x + int(overshoot_distance * math.cos(final_angle))
    overshoot_y = target_y + int(overshoot_distance * math.sin(final_angle))

    # Move to overshoot point first (only if overshoot is meaningful)
    if overshoot_percentage > 0:
        cursor.move_to([overshoot_x, overshoot_y], duration=0.08)
        time.sleep(0.02)  # Brief pause

    # Then move to final target
    cursor.move_to([target_x, target_y], duration=0.1)

    logger.debug(f"Moved mouse from ({current_x}, {current_y}) -> overshoot ({overshoot_x}, {overshoot_y}) -> target ({target_x}, {target_y})")


if __name__ == "__main__":
    # move_mouse_to_rect((-1612,282,-612,882))
    # x1,y1,w,h
    move_mouse_to_rect((-162,538,132,113), angle_variation_degrees=10.0, overshoot_percentage=20.0)