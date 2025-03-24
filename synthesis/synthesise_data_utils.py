import math
from data.ogham import Ogham 
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import random
def interpolate_x(points, y_target):
    """
    Finds the x value for a given y using linear interpolation.
    :param points: List of (x, y) tuples sorted by y values.
    :param y_target: The y value to find the corresponding x for.
    :return: Interpolated x value or None if out of range.
    """
    for i in range(len(points) - 1):
        (x1, y1), (x2, y2) = points[i], points[i + 1]
        
        # Check if y_target is within this segment
        if y1 <= y_target <= y2 or y2 <= y_target <= y1:
            # Avoid division by zero if x1 == x2 (vertical line segment)
            if y2 == y1:
                return x1
            
            # Apply interpolation formula
            x_target = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
            return x_target
    
    return None  # y_target is outside the given range

def make_vertical_and_center(line, canvas_height=300, canvas_width=300):
    middle_x = canvas_height // 2
    middle_y = canvas_width //2 
    vertical_line = [(y + middle_x, x+20) for x, y in line]
    return vertical_line

def get_joints(character:Ogham, points, canvas_height=300):
        new_points = []
        r = random.random()
        if r < 0.33:
            start = canvas_height // 5
            multiplier = 1
        elif r < 0.66:
            start = canvas_height // 3
            multiplier = 1
        else:
            start = canvas_height * 4/5
            multiplier = -1

        spacing = 40
        target_ys = [start + (multiplier * y) * spacing for y in range(character.lines)]
        # target_ys = [canvas_height // 5]
        # for y in range(1, character.lines):
        #     previous_y = target_ys[y-1]
        #     target_ys.append(previous_y + random.randrange(100,220,10) // character.lines)

        for target_y in target_ys:
            x_res = interpolate_x(points, target_y)
            new_points.append((x_res,target_y))
        return new_points

def get_length_of_line(points):
    total_length = 0.0

    # Iterate through the points and calculate the distance between consecutive points
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_length += distance

    return total_length

def scale_down_points(points, max_value=150):
    # Find the maximum coordinate value (x or y) in the points
    max_coordinate = max(max(abs(x), abs(y)) for x, y in points)

    # Calculate the scaling factor
    scaling_factor = max_value / max_coordinate

    # Scale down all the points
    scaled_points = [(x * scaling_factor, y * scaling_factor) for x, y in points]
    return scaled_points

def move_points_to_target(points, target_joint):
    # Extract the first point and the target joint
    first_point = points[0]
    target_x, target_y = target_joint

    # Calculate the difference in x and y coordinates
    delta_x = target_x - first_point[0]
    delta_y = target_y - first_point[1]

    # Move all points by delta_x and delta_y
    new_points = [(x + delta_x, y + delta_y) for x, y in points]

    return new_points

def rotate_point(x, y, origin_x, origin_y, angle):
    """Rotate a point around a given origin. Angle is in degrees."""
    angle_rad = math.radians(angle)
    x_rot = origin_x + (x - origin_x) * math.cos(angle_rad) - (y - origin_y) * math.sin(angle_rad)
    y_rot = origin_y + (x - origin_x) * math.sin(angle_rad) + (y - origin_y) * math.cos(angle_rad)
    return x_rot, y_rot


def add_border_and_resize(image:Image, width=300,height=300):
    image_with_border = Image.new("RGB", (width,height), color="white")
    x_offset = (width-image.width) // 2
    image_with_border.paste(image, (x_offset, 0))
    return image_with_border


def zoom_in(image, zoom_factor=1.2):
    width, height = image.size
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)

    # Crop center
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    cropped_image = image.crop((left, top, right, bottom))  # Crop to new size
    return cropped_image.resize((width, height), Image.LANCZOS)  # Resize back to original


def zoom_out(image, zoom_factor=1.3):
    width, height = image.size
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)

    # Resize smaller
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create blank canvas and paste resized image
    new_image = Image.new("RGB", (width, height), (255,255,255))  # Black background
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2
    new_image.paste(resized_image, (x_offset, y_offset))

    return new_image


def crop_center(image, target_size):
    width, height = image.size
    new_width, new_height = target_size

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))


def randomise_location(image):
    background = Image.new('RGB', (300, 300), (255, 255, 255))
    x_offset = np.random.randint(-60, 60)
    y_offset = np.random.randint(-50, 25)

    # Paste the original image onto the background at the random offset
    background.paste(image, (x_offset, y_offset))
    return background