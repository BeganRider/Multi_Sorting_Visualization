import pygame
import random
import math
import time

pygame.init()


class DrawInformation:
    BLACK = 0, 0, 0
    WHITE = 255, 255, 255
    GREEN = 0, 255, 0
    RED = 255, 0, 0
    BACKGROUND_COLOR = WHITE

    GRADIENTS = [
        (128, 128, 128),
        (160, 160, 160),
        (192, 192, 192)
    ]

    FONT = pygame.font.SysFont('comicsans', 24)
    LARGE_FONT = pygame.font.SysFont('comicsans', 32)

    SIDE_PAD = 100
    TOP_PAD = 120

    def __init__(self, window, x_offset, width, height, lst):
        self.window = window
        self.x_offset = x_offset
        self.width = width
        self.height = height
        self.set_list(lst)

    def set_list(self, lst):
        self.lst = lst
        self.min_val = min(lst)
        self.max_val = max(lst)

        self.block_width = round((self.width - self.SIDE_PAD) / len(lst))
        self.block_height = math.floor((self.height - self.TOP_PAD) / (self.max_val - self.min_val))
        self.start_x = self.x_offset + self.SIDE_PAD // 2


def draw(draw_info, algo_name, ascending, step_count, time):

    draw_info.window.fill(draw_info.BACKGROUND_COLOR, (draw_info.x_offset, 0, draw_info.width, draw_info.height))

    title = draw_info.LARGE_FONT.render(f"{algo_name} - {'Asc' if ascending else 'Desc'}", 1, draw_info.GREEN)
    draw_info.window.blit(title, (draw_info.x_offset + draw_info.width/2 - title.get_width()/2, 5))

    steps = draw_info.FONT.render(f"Steps: {step_count}", 1, draw_info.BLACK)
    draw_info.window.blit(steps, (draw_info.x_offset + draw_info.width/2 - steps.get_width()/2, 40))

    time_display = draw_info.FONT.render(f"Time: {time:.2f}s", 1, draw_info.BLACK)
    draw_info.window.blit(time_display, (draw_info.x_offset + draw_info.width/ 2 - time_display.get_width() / 2, 80))

    draw_list(draw_info)
    pygame.display.update()


def draw_list(draw_info, color_positions={}, clear_bg=False):
    lst = draw_info.lst

    if clear_bg:
        clear_rect = (draw_info.x_offset + draw_info.SIDE_PAD // 2, draw_info.TOP_PAD,
                      draw_info.width - draw_info.SIDE_PAD, draw_info.height - draw_info.TOP_PAD)
        pygame.draw.rect(draw_info.window, draw_info.BACKGROUND_COLOR, clear_rect)

    for i, val in enumerate(lst):
        x = draw_info.start_x + i * draw_info.block_width
        y = draw_info.height - (val - draw_info.min_val) * draw_info.block_height

        color = draw_info.GRADIENTS[i % 3]

        if i in color_positions:
            color = color_positions[i]

        pygame.draw.rect(draw_info.window, color, (x, y, draw_info.block_width, draw_info.height))

    if clear_bg:
        pygame.display.update()


def generate_starting_list(n, min_val, max_val):
    return [random.randint(min_val, max_val) for _ in range(n)]


def bubble_sort(draw_info, ascending=True):
    lst = draw_info.lst
    steps = 0

    for i in range(len(lst) - 1):
        steps += 1  # outer loop iteration
        for j in range(len(lst) - 1 - i):
            steps += 1  # inner loop iteration
            num1 = lst[j]
            num2 = lst[j + 1]

            if (num1 > num2 and ascending) or (num1 < num2 and not ascending):
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                draw_list(draw_info, {j: draw_info.GREEN, j + 1: draw_info.RED}, True)
            yield steps

    yield steps


def insertion_sort(draw_info, ascending=True):
    lst = draw_info.lst
    steps = 0

    for i in range(1, len(lst)):
        steps += 1  # outer loop
        current = lst[i]
        j = i
        while j > 0 and ((lst[j - 1] > current and ascending) or (lst[j - 1] < current and not ascending)):
            steps += 1  # each time the while loop runs
            lst[j] = lst[j - 1]
            j -= 1
            lst[j] = current
            draw_list(draw_info, {j - 1: draw_info.GREEN, j: draw_info.RED}, True)
            yield steps
    yield steps


def selection_sort(draw_info, ascending=True):
    lst = draw_info.lst
    steps = 0

    for i in range(len(lst)):
        steps += 1  # outer loop
        min_index = i
        for j in range(i + 1, len(lst)):
            steps += 1  # inner loop
            if (lst[j] < lst[min_index] and ascending) or (lst[j] > lst[min_index] and not ascending):
                min_index = j

            draw_list(draw_info, {j: draw_info.GREEN, min_index: draw_info.RED}, True)
            yield steps  # yield after every comparison to show progress

        if min_index != i:
            lst[i], lst[min_index] = lst[min_index], lst[i]
            draw_list(draw_info, {i: draw_info.GREEN, min_index: draw_info.RED}, True)

    yield steps

def quicksort_sort(draw_info, ascending=True):
    lst = draw_info.lst
    steps = 0
    stack = [(0, len(lst) - 1)]

    def partition(low, high):
        nonlocal steps
        pivot = lst[high]
        i = low - 1
        for j in range(low, high):
            steps += 1  # inner loop iteration
            if (lst[j] <= pivot and ascending) or (lst[j] >= pivot and not ascending):
                i += 1
                lst[i], lst[j] = lst[j], lst[i]
                draw_list(draw_info, {i: draw_info.GREEN, j: draw_info.RED}, True)
                yield steps
        lst[i + 1], lst[high] = lst[high], lst[i + 1]
        draw_list(draw_info, {i + 1: draw_info.GREEN, high: draw_info.RED}, True)
        yield steps
        return i + 1

    while stack:
        steps += 1  # outer loop (partitioning region)
        low, high = stack.pop()
        if low < high:
            gen = partition(low, high)
            while True:
                try:
                    yield next(gen)
                except StopIteration as e:
                    pivot_index = e.value if e.value is not None else (low + high) // 2
                    break
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))

    yield steps







def main():
    run = True
    clock = pygame.time.Clock()

    window_width = 1200
    window_height = 700
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Side-by-Side Sorting Algorithms")

    n = 50
    min_val = 0
    max_val = 100
    starting_list = generate_starting_list(n, min_val, max_val)

    half_width = window_width // 2
    quart_width = half_width // 2
    bubble_info = DrawInformation(window, 0, quart_width, window_height, starting_list.copy())
    insertion_info = DrawInformation(window, quart_width, quart_width, window_height, starting_list.copy())
    selection_info = DrawInformation(window, half_width, quart_width, window_height, starting_list.copy())
    quicksort_info = DrawInformation(window, half_width + quart_width, quart_width, window_height, starting_list.copy())

    selection_selction_gen = selection_sort(selection_info, ascending=True)
    quicksort_sort_gen = quicksort_sort(quicksort_info, ascending=True)
    bubble_sort_gen = bubble_sort(bubble_info, ascending=True)
    insertion_sort_gen = insertion_sort(insertion_info, ascending=True)


    sorting = False

    bubble_steps = 0
    insertion_steps = 0
    selection_steps = 0
    quicksort_steps = 0

    bubble_start_time = 0
    bubble_elapsed_time = 0
    insertion_start_time = 0
    insertion_elapsed_time = 0
    selection_start_time = 0
    selection_elapsed_time = 0
    quicksort_start_time = 0
    quicksort_elapsed_time = 0

    bubble_sorting = True
    insertion_sorting = True
    selection_sorting = True
    quicksort_sorting = True

    while run:
        clock.tick(120)

        if sorting == True:
            if bubble_sorting:
                try:
                    bubble_steps = next(bubble_sort_gen)
                    bubble_elapsed_time = time.time() - bubble_start_time
                except StopIteration:
                    bubble_sorting = False
                    bubble_elapsed_time = time.time() - bubble_start_time

            if insertion_sorting:
                try:
                   insertion_steps = next(insertion_sort_gen)
                   insertion_elapsed_time = time.time() - insertion_start_time
                except StopIteration:
                    insertion_sorting = False

            if selection_sorting:
                try:
                   selection_steps = next(selection_selction_gen)
                   selection_elapsed_time = time.time() - selection_start_time
                except StopIteration:
                    selection_sorting = False

            if quicksort_sorting:
                try:
                   quicksort_steps = next(quicksort_sort_gen)
                   quicksort_elapsed_time = time.time() - quicksort_start_time
                except StopIteration:
                    quicksort_sorting = False

        draw(bubble_info, "Bubble Sort", True, bubble_steps, bubble_elapsed_time)
        draw(insertion_info, "Insertion Sort", True, insertion_steps, insertion_elapsed_time)
        draw(quicksort_info, "Quick Sort", True, quicksort_steps, quicksort_elapsed_time)
        draw(selection_info, "Selection Sort", True, selection_steps, selection_elapsed_time)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_r:
                sorting = False
                bubble_elapsed_time=0
                insertion_elapsed_time=0
                selection_elapsed_time=0
                quicksort_elapsed_time=0
                lst = generate_starting_list(n, min_val, max_val)
                draw_info.set_list(lst)
            elif event.key == pygame.K_SPACE and sorting == False:
                sorting = True
                bubble_elapsed_time=0
                insertion_elapsed_time=0
                selection_elapsed_time=0
                quicksort_elapsed_time=0
                bubble_start_time=time.time()
                insertion_start_time=time.time()
                quicksort_start_time=time.time()
                selection_start_time=time.time()

    pygame.quit()


if __name__ == "__main__":
    main()