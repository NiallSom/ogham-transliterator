import random

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import os
import json
import math
import csv
from data.ogham import Ogham, Direction, Characters
from synthesise_data_utils import *

output_dir = "output/temp-letters-dir/"


class LineGenerator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.file_path, 'r') as file:
                return [json.loads(line) for line in file]
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return []

    def get_line(self):
        if not self.data:
            return []
        index = np.random.randint(len(self.data))
        drawing = self.data[index]['drawing']
        line = [(x, y) for x, y in zip(drawing[0][0], drawing[0][1])]
        if line[0][0] > line[-1][0]:
            line.reverse()
        return line


class OghamGenerator:
    def __init__(self, canvas_size=(200, 300)):
        self.canvas_size = canvas_size
        self.line_generator = LineGenerator("synthesis/data/lines.ndjson")
        self.image = None
        self.canvas = None
        self.joints = []

    def _draw_vertical_line(self, for_char):
        self.image = Image.new("RGB", self.canvas_size, "white")
        self.canvas = ImageDraw.Draw(self.image)
        horizontal_line = self.line_generator.get_line()
        vertical_line = make_vertical_and_center(horizontal_line, self.canvas_size[0], self.canvas_size[1])
        self.joints = get_joints(for_char, vertical_line)
        self.canvas.line(vertical_line, fill="black", width=5)

    def _handle_horizontal(self, line, joint):
        new_points = [(x + joint[0] // 6, joint[1] - y) for x, y in line]
        return new_points

    def _handle_right(self, line, joint):
        new_points = [(x + joint[0] // 6, joint[1] - y) for x, y in line]
        new_points = move_points_to_target(new_points, joint)
        return new_points

    def _handle_left(self, line, joint):
        new_points = [(x + joint[0] // 6, joint[1] - y) for x, y in line]
        new_points.reverse()
        new_points = move_points_to_target(new_points, joint)
        return new_points

    def _handle_diagonal(self, line, joint, angle):
        new_points = [(x + joint[0] // 6, joint[1] - y) for x, y in line]
        rot_points = [rotate_point(x - 20, y + 40, 150, y, angle) for x, y in new_points]
        return rot_points

    def draw_character(self, character: Ogham, random_width=False):
        self._draw_vertical_line(character)
        angle = random.randrange(30, 60)
        direction_handlers = {
            Direction.HORIZONTAL: lambda line, joint: self._handle_horizontal(line, joint),
            Direction.RIGHT: lambda line, joint: self._handle_right(scale_down_points(line, 150), joint),
            Direction.LEFT: lambda line, joint: self._handle_left(scale_down_points(line, 150), joint),
            Direction.DIAGONAL: lambda line, joint: self._handle_diagonal(line, joint, angle)
        }

        for joint in self.joints:
            # self.canvas.ellipse((joint[0]-5,joint[1]-5,joint[0]+5, joint[1]+5),fill="red")
            horizontal_line = self.line_generator.get_line()
            handler = direction_handlers[character.direction](horizontal_line, joint)
            self.canvas.line(handler, fill="black", width=7 if not random_width else np.random.randint(4, 7))

        return add_border_and_resize(self.image)


def generate_dataset(dataset_size=10, normal=0.4, skewed=0.3, scaled=0.3):
    os.makedirs(output_dir, exist_ok=True)
    generator = OghamGenerator()
    count = 0
    num_normal = int(dataset_size * normal)
    num_skewed = int(dataset_size * skewed)
    num_scaled = int(dataset_size * scaled)
    with open("output/labels.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["img_no", "img_letter", "type"])
        for character in Characters:
            for _ in range(num_normal):
                image = generator.draw_character(character.value, random_width=True)
                image = randomise_location(image=image)
                padded_count = f"{count:05}"
                image.save(f"{output_dir}{padded_count}.png")
                writer.writerow([padded_count, character.name, "normal"])
                count += 1
            for _ in range(num_skewed):
                image = generator.draw_character(character.value, random_width=True)
                image = randomise_location(image=image)
                image = image.rotate(np.random.randint(-15, 15), expand=True, fillcolor="white")
                image = crop_center(image, (300, 300))
                padded_count = f"{count:05}"
                image.save(f"{output_dir}{padded_count}.png")
                writer.writerow([padded_count, character.name, "skewed"])
                count += 1
            for _ in range(num_scaled):
                image = generator.draw_character(character.value, random_width=True)
                image = zoom_in(image=image) if np.random.randint(0, 2) else zoom_out(image=image)
                image = randomise_location(image=image)
                padded_count = f"{count:05}"
                image.save(f"{output_dir}{padded_count}.png")
                writer.writerow([padded_count, character.name, "scaled"])
                count += 1


def main():
    generate_dataset()
    # os.makedirs(output_dir, exist_ok=True)
    # generator = OghamGenerator()
    # image = generator.draw_character(Characters.D.value,random_width=True)
    # image = randomise_location(image=image)
    # image.save(f"L.png")


if __name__ == "__main__":
    main()
