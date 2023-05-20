import numpy as np
import matplotlib.pyplot as plt


class Rectangle:
    """A class representing a single rectangle in the cutting stock problem.
    Each rectangle has its ID, height, width and value."""
    __slots__ = ['height', 'width', 'value']

    def __init__(self, height: int, width: int, value: int):
        self.height = height
        self.width = width
        self.value = value

    def __str__(self):
        return f'Rectangle(width={self.width}, height={self.height}, value={self.value})'

    def __repr__(self):
        return str(self)


class Circle:
    """A class representing an individual of the
    population in the genetic algorithm"""
    __slots__ = ['radius', 'rows']

    def __init__(self, radius: float, rows: list[Rectangle] = None):
        """
        :param radius:  circle radius
        :param rows:    rectangles in rows, from bottom to top
        """
        if rows is None:
            rows = []
        self.radius = radius
        self.rows = rows

    def __str__(self):
        result = f'Circle(radius={self.radius}, rows=['
        for rect, row_length in zip(self.rows, self.get_rows_lengths()):
            result += f'\n\t<{rect}, length={row_length}>,'
        result += '\n])'
        return result

    def __repr__(self):
        return str(self)

    @property
    def bottom_gap(self) -> float:
        """Distance between the base of the bottommost rectangle and the
        lowest point of the circle"""
        bottom_rect_half_width = self.rows[0].width / 2
        return self.radius - np.sqrt(self.radius ** 2 - bottom_rect_half_width ** 2)

    def score(self) -> int:
        """Calculates the score of this circle"""
        score = 0
        for rect, row_length in zip(self.rows, self.get_rows_lengths()):
            score += rect.value * row_length
        return score

    def get_rows_lengths(self) -> list[int]:
        """Calculates the number of rectangles in each row"""
        # the bottom row has a length of 1
        rows_lengths = [1]
        # curr_height - the height at which the row we currently look at starts
        # can range from 0 to 2*radius
        curr_height = self.bottom_gap + self.rows[0].height
        for rect in self.rows[1:]:
            if curr_height + rect.height >= 2 * self.radius:  # overflow
                remaining_lengths = [0 for _ in range(len(self.rows) - len(rows_lengths))]
                rows_lengths.extend(remaining_lengths)
                break
            # available width at the height of the bottom of the current rectangle
            # and at the height of its top
            avail_width_bottom = 2 * np.sqrt(
                self.radius ** 2 - (curr_height - self.radius) ** 2
            )
            avail_width_top = 2 * np.sqrt(
                self.radius ** 2 - (curr_height + rect.height - self.radius) ** 2
            )
            max_row_width = np.min([avail_width_bottom, avail_width_top])
            # how many recangles will fit
            row_length = int(np.floor(max_row_width / rect.width))
            rows_lengths.append(row_length)
            # update curr_height
            curr_height += rect.height
        return rows_lengths

    def is_overflow(self) -> bool:
        return self.get_rows_lengths()[-1] == 0

    def trim(self):
        """Removes empty rows from the top"""
        row_lengths = self.get_rows_lengths()
        # find the first non-zero element starting from the end of the list
        last_zero_idx = 0
        for i in range(-1, -len(self.rows) - 1, -1):
            if row_lengths[i] != 0:
                last_zero_idx = i + 1
                break
        if last_zero_idx != 0:  # empty rows found
            self.rows = self.rows[:last_zero_idx]

    def plot(self, detailed=True):
        """
        Visualises the circle

        :param detailed:    whether or not to display rectangles details
        """
        ax: plt.Axes = plt.gca()
        circle = plt.Circle(
            (0, self.radius), radius=self.radius,
            facecolor='none', edgecolor='black',
        )
        ax.add_patch(circle)

        curr_height = self.bottom_gap
        for rect, row_length in zip(self.rows, self.get_rows_lengths()):
            # x coordinate of the left-bottom corner of
            # the leftmost rectangle in a row
            x_coord = np.max([
                -np.sqrt(self.radius ** 2 - (curr_height - self.radius) ** 2),
                -np.sqrt(self.radius ** 2 - (curr_height + rect.height - self.radius) ** 2),
            ])
            for i in range(row_length):
                # x coordinate of the left-bottom corner of
                # the current rectangle
                x = x_coord + i * rect.width
                new_rect = plt.Rectangle(
                    (x, curr_height), width=rect.width, height=rect.height,
                    facecolor='lightblue', edgecolor='black',
                )
                ax.add_patch(new_rect)
            if detailed:
                # add a label with the rectangle info
                text_x = self.radius + 50
                text_y = curr_height + rect.height / 2 - 25
                plt.text(text_x, text_y, str(rect))
            # update curr_height
            curr_height += rect.height
        plt.axis('scaled')
        plt.axis('off')
        plt.title(f'Radius: {self.radius}   |   Score: {self.score()}')
        plt.show()
