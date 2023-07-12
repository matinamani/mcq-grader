import cv2 as cv
import numpy as np
import pandas as pd

from constants import BLANK_WIDTH, COLUMN_WIDTH, KEY, ROOT


class Grader:
    def __init__(self, image_paths):
        self.load_images(image_paths)
        self.preprocess()
        self.extract_cells()
        self.process_answers()

    def load_images(self, paths):
        self.images = []
        for path in paths:
            self.images.append(
                {
                    "id": path.replace("/image", "").replace(".tif", ""),
                    "raw": self.resize_and_crop(path),
                }
            )

    def preprocess(self):
        for image in self.images:
            image.update(
                {
                    "raw": cv.threshold(
                        image["raw"], 190, 255, cv.THRESH_BINARY_INV
                    )[1]
                }
            )

    def extract_cells(self):
        for image in self.images:
            image.update({"cells": self.crop_cells(image["raw"])})

    def process_answers(self):
        for image in self.images:
            image.update({"answers": self.get_answers(image)})
            self.calculate_grade(image)

    def get_answers(self, image):
        answers = []
        for cell in image["cells"]:
            answers += self.get_cell_answers(cell)

        del answers[148]

        return answers[:165]

    def get_cell_answers(self, cell):
        cell_answers = []
        rows = np.vsplit(self.adjust_height(cell, 10), 10)
        for row in rows:
            cell_answers.append(self.determine_bubble(row))

        return cell_answers

    def determine_bubble(self, row):
        boxes = np.hsplit(self.adjust_width(row, 5), 5)[1:]
        box_pixel_values = []
        for box in boxes:
            box_pixel_values.append(cv.countNonZero(box))

        if max(box_pixel_values) >= 65:
            return np.argmax(box_pixel_values) + 1

        return 0

    def calculate_grade(self, image):
        correct_answers = 0
        for x, y in zip(image["answers"], KEY):
            if x == y:
                correct_answers += 1

        image.update({"score": (correct_answers / len(KEY)) * 100})

    def crop_cells(self, image):
        cells = []
        columns = self.extract_columns(image)
        for column in columns:
            cells += np.vsplit(self.adjust_height(column[5:, :], 5), 5)

        return cells

    def extract_columns(self, image):
        return [
            image[:, :COLUMN_WIDTH],
            image[
                :, COLUMN_WIDTH + BLANK_WIDTH : (2 * COLUMN_WIDTH) + BLANK_WIDTH
            ],
            image[
                :,
                (2 * COLUMN_WIDTH)
                + (2 * BLANK_WIDTH) : (3 * COLUMN_WIDTH)
                + (2 * BLANK_WIDTH),
            ],
            image[:, (3 * COLUMN_WIDTH) + (3 * BLANK_WIDTH) :],
        ]

    def adjust_height(self, image, number_of_slices):
        image_height = image.shape[0]
        split_height, reminder = divmod(image_height, number_of_slices)

        if reminder != 0:
            adjusted_height = image_height - reminder
            return image[:adjusted_height, :]
        return image

    def adjust_width(self, image, number_of_slices):
        image_width = image.shape[1]
        split_width, reminder = divmod(image_width, number_of_slices)

        if reminder != 0:
            adjusted_width = image_width - reminder
            return image[:, :adjusted_width]
        return image

    def show_images(self):
        for image in self.images:
            cv.imshow(image["id"], image["raw"])

        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_image(self, index):
        index %= len(self.images)
        cv.imshow(self.images[index]["id"], self.images[index]["raw"])
        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_cells(self, index):
        index %= len(self.images)
        for i, cell in enumerate(self.images[index]["cells"]):
            cv.imshow(f'{self.images[index]["id"]}_{i}', cell)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def resize_and_crop(self, path):
        img = cv.imread(ROOT + path)
        new_height = int(img.shape[1] * 0.4)
        new_width = int(img.shape[0] * 0.4)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (new_height, new_width))
        return img[245:-85, 35:-230]

    def show_answers(self, index):
        index %= len(self.images)
        for i, answer in enumerate(self.images[index]["answers"]):
            print(f"{i+1}: {answer}")

    def show_scores(self):
        for image in self.images:
            print(f"{image['id']}: {image['score']}%")

    def show_score(self, index):
        index %= len(self.images)
        print(f"{self.images[index]['id']}: {self.images[index]['score']}%")

    def save_status(self, index):
        index %= len(self.images)
        data = {
            "questions": self.generate_question_indices(),
            "status": self.get_status(self.images[index]["answers"]),
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.images[index]['id']}.csv", index=False)

    def save_all_status(self):
        ids = [image["id"] for image in self.images]
        df = pd.DataFrame(columns=["questions"] + ids)

        df["questions"] = self.generate_question_indices()
        for image in self.images:
            df[image["id"]] = self.get_status(image["answers"])

        df.to_csv("all_status.csv", index=False)

    def generate_question_indices(self):
        question_arr = np.arange(165) + 1
        return np.concatenate([question_arr[:148], question_arr[149:]])

    def get_status(self, answers):
        status = []
        for x, y in zip(answers, KEY):
            if x == y:
                status.append(True)
            elif x == 0:
                status.append("-")
            else:
                status.append(False)

        return status
