import matplotlib.pyplot as plt



def bresenham_line(start, goal):
        """Bresenham's line algorithm
        Args:
            start: (float64, float64) - start coordinate
            goal: (float64, float64) - goal coordinate
        Returns:
            interlying points between the start and goal coordinate
        """
        # Extract scalar values from numpy arrays
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(goal[0]), int(goal[1])
        
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        # Add the final goal point
        # line.append((x1, y1))
        # Add the final goal point
        # line.append((x1, y1))
        x = goal[0]
        y = goal[1]
        # Remove the start point if it is in the line
        if (x0, y0) in line:
            line.remove((x0, y0))
    
        return line

def dda_line(start, goal):
    """DDA (Digital Differential Analyzer) line algorithm
    Args:
        start: (float64, float64) - start coordinate
        goal: (float64, float64) - goal coordinate
    Returns:
        List of points between the start and goal coordinates
    """
    # Extract scalar values from numpy arrays
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(goal[0]), float(goal[1])
    
    line = []
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))  # Number of steps needed

    if steps == 0:
        return [(int(x0), int(y0))]  # Single point case
    
    x_step = dx / steps
    y_step = dy / steps
    
    x, y = x0, y0
    for _ in range(steps):
        line.append((int(round(x)), int(round(y))))  # Round to nearest integer
        x += x_step
        y += y_step

    return line


def bresenham_circle(x0, y0, radius):
    x = radius
    y = 0
    err = 0

    points = []

    while x >= y:
        # Calculate the points for all eight octants
        points.extend([
            (x0 + x, y0 + y),
            (x0 + y, y0 + x),
            (x0 - y, y0 + x),
            (x0 - x, y0 + y),
            (x0 - x, y0 - y),
            (x0 - y, y0 - x),
            (x0 + y, y0 - x),
            (x0 + x, y0 - y)
        ])

        y += 1
        err += 1 + 2 * y
        if 2 * (err - x) + 1 > 0:
            x -= 1
            err += 1 - 2 * x

    return points

def bresenham_circle_segment(x0, y0, radius):
    x = radius
    y = 0
    err = 0

    points = []

    while x >= y:
        # Only generate points in the range -pi/6 to +pi/6
        if y <= x * (1 / (3 ** 0.5)):  # This condition ensures we stay within the -pi/6 to +pi/6 range
            points.append((x0 + x, y0 + y))  # First octant
            points.append((x0 + x, y0 - y))  # Eighth octant

        y += 1
        err += 1 + 2 * y
        if 2 * (err - x) + 1 > 0:
            x -= 1
            err += 1 - 2 * x

    return points


class WuLineAlgorithm:
    def __init__(self, point1, point2):
        self.x1, self.y1 = point1
        self.x2, self.y2 = point2
        self.dx, self.dy = self.x2 - self.x1, self.y2 - self.y1
        self.steep = abs(self.dy) > abs(self.dx)
        self.calc_details()
        self.startX, self.endX = self.calc_endPoint(point1) + 1, self.calc_endPoint(point2)
        self.points = self.calculate_points()

    def calc_points(self, xCoOrd, yCoOrd):
        return ((xCoOrd, yCoOrd), (yCoOrd, xCoOrd))[self.steep]

    def fPart(self, x):
        return x - int(x)

    def rfPart(self, x):
        return 1 - self.fPart(x)

    def calc_endPoint(self, point):
        x, y = point
        return int(round(x))

    def calc_details(self):
        if self.steep:
            self.x1, self.x2, self.y1, self.y2, self.dx, self.dy = self.y1, self.y2, self.x1, self.x2, self.dy, self.dx
        if self.x2 < self.x1:
            self.x1, self.x2, self.y1, self.y2 = self.x2, self.x1, self.y2, self.y1
        self.gradient = self.dy / self.dx
        self.yIntersection = self.y1 + self.rfPart(self.x1) * self.gradient

    def calculate_points(self):
        points = []
        for x in range(self.startX, self.endX):
            y = int(self.yIntersection)
            points.append(self.calc_points(x, y))  # Main pixel
            points.append(self.calc_points(x, y + 1))  # Upper pixel
            points.append(self.calc_points(x, y - 1))  # Lower pixel (helps fill gaps)
            self.yIntersection += self.gradient
        return points


    def get_points(self):
        return self.points
    
    