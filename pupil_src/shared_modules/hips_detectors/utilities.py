def degree_arc_of_raster_circle(circle_array):
    return 360 / len(circle_array)


def create_circle(radius):
    """
    Bresenham’s Circle Drawing Algorithm.
    :param radius:
    :return:
    """
    border_amount = 1
    switch = 3 - (2 * radius)
    x = 0
    y = radius

    one = []
    two = []
    three = []
    four = []
    five = []
    six = []
    seven = []
    eight = []

    # first quarter/octant starts clockwise at 12 o'clock
    one.append(get_array_index_from_coords(-y, -x, radius, border_amount))
    three.append(get_array_index_from_coords(x, -y, radius, border_amount))
    five.append(get_array_index_from_coords(y, x, radius, border_amount))
    seven.append(get_array_index_from_coords(-x, y, radius, border_amount))

    while x <= y:
        x += 1
        if switch < 0:
            switch = switch + (4 * x) + 6
        else:
            switch = switch + (4 * (x - y)) + 10
            y -= 1

        one.append(get_array_index_from_coords(-y, -x, radius, border_amount))
        two.append(get_array_index_from_coords(-x, -y, radius, border_amount))
        three.append(get_array_index_from_coords(x, -y, radius, border_amount))
        four.append(get_array_index_from_coords(y, -x, radius, border_amount))
        five.append(get_array_index_from_coords(y, x, radius, border_amount))
        six.append(get_array_index_from_coords(x, y, radius, border_amount))
        seven.append(get_array_index_from_coords(-x, y, radius, border_amount))
        eight.append(get_array_index_from_coords(-y, x, radius, border_amount))

    all = []
    all.extend(one)
    all.extend(two)
    all.extend(three)
    all.extend(four)
    all.extend(five)
    all.extend(six)
    all.extend(seven)
    all.extend(eight)

    return all


def get_array_index_from_coords(x, y, radius, border_amount):
    # Translate point so that circle lies entirely in the 1st quadrant
    x += radius + border_amount
    y += radius + border_amount

    width = (radius * 2) + (border_amount * 2)
    y = width - y
    index = (y * width) + x

    return index

