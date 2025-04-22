import pygame

import math
import time
import json
from csv_importer import load_datasets_from_csv, generate_starting_list, export_results_to_csv

import pandas as pd

dataframe = None
csv_loaded = False


filename = "sorted_lists.csv"
csv_data_columns = []
csv_data = None
csv_columns = []

pygame.init()

current_column = 0
results = []

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

    FONT = pygame.font.SysFont('comicsans', 22)
    LARGE_FONT = pygame.font.SysFont('comicsans', 32)

    SIDE_PAD = 100
    TOP_PAD = 200

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


def draw(draw_info, algo_name, ascending, step_count, time, ticks):

    draw_info.window.fill(draw_info.BACKGROUND_COLOR, (draw_info.x_offset, 0, draw_info.width, draw_info.height))

    title = draw_info.LARGE_FONT.render(f"{algo_name}", 1, draw_info.GREEN)
    draw_info.window.blit(title, (draw_info.x_offset + draw_info.width/2 - title.get_width()/2, 5))

    steps = draw_info.FONT.render(f"Swaps: {step_count}", 1, draw_info.BLACK)
    draw_info.window.blit(steps, (draw_info.x_offset + draw_info.width/2 - steps.get_width()/2, 40))

    time_display = draw_info.FONT.render(f"Time: {time:.2f}s", 1, draw_info.BLACK)
    draw_info.window.blit(time_display, (draw_info.x_offset + draw_info.width/ 2 - time_display.get_width() / 2, 80))

    ticks = draw_info.FONT.render(f"Refresh Ticks: {ticks}", 1, draw_info.BLACK)
    draw_info.window.blit(ticks, (draw_info.x_offset + draw_info.width / 2 - ticks.get_width() / 2, 60))

    draw_list(draw_info)
    pygame.display.update()

def load_csv_dataset(path):
    global dataframe
    dataframe = pd.read_csv(path)
    return dataframe

def get_column_data(index):
    global dataframe
    if index >= len(dataframe.columns):
        return None
    column = dataframe.iloc[:, index].dropna().tolist()
    return [int(val) for val in column]

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


def bubble_sort(draw_info, ascending=True):
    lst = draw_info.lst
    steps = 0

    for i in range(len(lst) - 1):
        swapped = False  # Flag to track if any swap occurred

        for j in range(len(lst) - 1 - i):
            num1 = lst[j]
            num2 = lst[j + 1]

            if (num1 > num2 and ascending) or (num1 < num2 and not ascending):
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                draw_list(draw_info, {j: draw_info.GREEN, j + 1: draw_info.RED}, True)
                steps += 1  # count a swap as a step
                swapped = True  # A swap occurred
            yield steps

        # If no swaps occurred, the list is sorted
        if not swapped:
            break  # Stop early as the list is sorted

    yield steps


def selection_sort(draw_info, ascending=True):
    lst = draw_info.lst
    steps = 0

    for i in range(len(lst)):
        min_index = i
        for j in range(i + 1, len(lst)):
            if (lst[j] < lst[min_index] and ascending) or (lst[j] > lst[min_index] and not ascending):
                min_index = j
            draw_list(draw_info, {j: draw_info.GREEN, min_index: draw_info.RED}, True)
            yield steps

        if min_index != i:
            lst[i], lst[min_index] = lst[min_index], lst[i]
            draw_list(draw_info, {i: draw_info.GREEN, min_index: draw_info.RED}, True)
            steps += 1

    yield steps


def insertion_sort(draw_info, ascending=True):
    lst = draw_info.lst
    steps = 0

    for i in range(1, len(lst)):
        current = lst[i]
        j = i
        while j > 0 and ((lst[j - 1] > current and ascending) or (lst[j - 1] < current and not ascending)):
            lst[j] = lst[j - 1]
            j -= 1
            lst[j] = current
            draw_list(draw_info, {j - 1: draw_info.GREEN, j: draw_info.RED}, True)
            steps += 1  # count each shift as a step
            yield steps
        # Only count a swap if an actual insertion happens
        if j != i:
            steps += 1  # count the final placement of the current element as a swap
            yield steps
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
            if (lst[j] <= pivot and ascending) or (lst[j] >= pivot and not ascending):
                i += 1
                lst[i], lst[j] = lst[j], lst[i]
                draw_list(draw_info, {i: draw_info.GREEN, j: draw_info.RED}, True)
                steps += 1  # count a swap as a step
            yield steps
        lst[i + 1], lst[high] = lst[high], lst[i + 1]
        draw_list(draw_info, {i + 1: draw_info.GREEN, high: draw_info.RED}, True)
        steps += 1  # count the swap of pivot placement
        yield steps
        return i + 1

    while stack:
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

def show_dataset_selection_screen(window):
    font = pygame.font.SysFont('comicsans', 32)
    small_font = pygame.font.SysFont('comicsans', 24)

    selecting = True
    while selecting:
        window.fill((255, 255, 255))

        title = font.render("Choose Data Source", True, (0, 0, 0))
        option1 = small_font.render("Press R for Random Dataset", True, (0, 0, 0))
        option2 = small_font.render("Press D to Load from CSV", True, (0, 0, 0))

        window.blit(title, (window.get_width() // 2 - title.get_width() // 2, 200))
        window.blit(option1, (window.get_width() // 2 - option1.get_width() // 2, 300))
        window.blit(option2, (window.get_width() // 2 - option2.get_width() // 2, 340))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "random"
                elif event.key == pygame.K_d:
                    return "csv"

def get_tick_speed_input():
    pygame.init()
    input_window = pygame.display.set_mode((600, 200))
    pygame.display.set_caption("Enter Tick Speed")
    font = pygame.font.SysFont("comicsans", 40)
    input_text = ""
    active = True

    while active:
        input_window.fill((255, 255, 255))
        prompt = font.render("Enter refresh tick speed (FPS):", True, (0, 0, 0))
        input_surface = font.render(input_text, True, (0, 0, 255))
        input_window.blit(prompt, (20, 30))
        input_window.blit(input_surface, (20, 100))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if input_text.isdigit():
                        return int(input_text)
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isdigit():
                    input_text += event.unicode

def print_controls():
    print("=== Controls ===")
    print("SPACE - Start Sorting")
    print("R - Reset")
    print("T - Change Tick Speed")
    print("A - Change Dataset Origin")
    print("================\n")

def export_results():
    if results:
        df = pd.DataFrame(results)
        df.to_csv("sorting_results.csv", index=False)
        print("Results exported to sorting_results.csv")

def load_config(path="settings.json"):
    with open(path, "r") as f:
        return json.load(f)

def main():
    current_column= 0
    results = []

    results = {
        "Bubble Sort": {},
        "Insertion Sort": {},
        "Selection Sort": {},
        "Quick Sort": {}
    }


    print_controls()
    tick_speed = get_tick_speed_input()
    run = True
    clock = pygame.time.Clock()
    run = True
    clock = pygame.time.Clock()

    config = load_config()

    window_width = config["window_width"]
    window_height = config["window_height"]
    n = config["n"]
    min_val = config["min_val"]
    max_val = config["max_val"]
 #   tick_speed = config["default_tick_speed"]

    #window_width = 1200
    #window_height = 600
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Side-by-Side Sorting Algorithms")

    #n = 50
    #min_val = 0
    #max_val = 100


    dataset_choice = show_dataset_selection_screen(window)

    if dataset_choice == "random":
        starting_list = generate_starting_list(n, min_val, max_val)
    elif dataset_choice == "csv":
        import pandas as pd
        #filename = "sorted_lists.csv"  # or prompt for file
        df = pd.read_csv(filename)
        # Use the first numeric column, for example:
        starting_list = df[df.columns[0]].dropna().astype(int).tolist()
        csv_columns = list(df.columns)

    half_width = window_width // 2
    quart_width = half_width // 2
    bubble_info = DrawInformation(window, 0, quart_width, window_height, starting_list.copy())
    insertion_info = DrawInformation(window, quart_width, quart_width, window_height, starting_list.copy())
    selection_info = DrawInformation(window, half_width, quart_width, window_height, starting_list.copy())
    quicksort_info = DrawInformation(window, half_width + quart_width, quart_width, window_height, starting_list.copy())

    selection_sort_gen = selection_sort(selection_info, ascending=True)
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

    bubble_tick=0
    insertion_tick=0
    selection_tick=0
    quicksort_tick=0

    while run:
        clock.tick(tick_speed)

        if sorting == True:

            if bubble_sorting:
                try:
                    bubble_steps = next(bubble_sort_gen)
                    bubble_tick += 1  # increment only if sorting is still running
                    bubble_elapsed_time = time.time() - bubble_start_time
                    #print(bubble_tick)

                except StopIteration:
                    bubble_tick += 1  # increment once even when sorting finishes
                    bubble_sorting = False
                    bubble_elapsed_time = time.time() - bubble_start_time
                    results["Bubble Sort"] = {
                        "Steps": bubble_steps,
                        "Time (s)": round(bubble_elapsed_time, 3)
                    }

            if insertion_sorting:
                try:
                   insertion_steps = next(insertion_sort_gen)
                   insertion_tick += 1
                   insertion_elapsed_time = time.time() - insertion_start_time
                except StopIteration:
                    insertion_tick += 1
                    insertion_sorting = False
                    results["Insertion Sort"] = {
                        "Steps": insertion_steps,
                        "Time (s)": round(insertion_elapsed_time, 3)
                    }

            if selection_sorting:
                try:
                    selection_steps = next(selection_sort_gen)
                    selection_tick += 1
                    selection_elapsed_time = time.time() - selection_start_time
                except StopIteration:
                    selection_tick += 1
                    selection_sorting = False
                    results["Selection Sort"] = {
                        "Steps": selection_steps,
                        "Time (s)": round(selection_elapsed_time, 3)
                    }

            if quicksort_sorting:
                try:
                   quicksort_steps = next(quicksort_sort_gen)
                   quicksort_tick += 1
                   quicksort_elapsed_time = time.time() - quicksort_start_time
                except StopIteration:
                    quicksort_tick += 1
                    quicksort_sorting = False
                    results["Quick Sort"] = {
                        "Steps": quicksort_steps,
                        "Time (s)": round(quicksort_elapsed_time, 3)
                    }
        if not bubble_sorting and not insertion_sorting and not selection_sorting and not quicksort_sorting:
            if csv_loaded:
                results.append({
                    "Column": current_column,
                    "Bubble_Steps": bubble_steps,
                    "Bubble_Time": bubble_elapsed_time,
                    "Insertion_Steps": insertion_steps,
                    "Insertion_Time": insertion_elapsed_time,
                    "Selection_Steps": selection_steps,
                    "Selection_Time": selection_elapsed_time,
                    "Quicksort_Steps": quicksort_steps,
                    "Quicksort_Time": quicksort_elapsed_time
                })
                sorting = False

        draw(bubble_info, "Bubble Sort", True, bubble_steps, bubble_elapsed_time,bubble_tick)
        draw(insertion_info, "Insertion Sort", True, insertion_steps, insertion_elapsed_time,insertion_tick)
        draw(quicksort_info, "Quick Sort", True, quicksort_steps, quicksort_elapsed_time,quicksort_tick)
        draw(selection_info, "Selection Sort", True, selection_steps, selection_elapsed_time,selection_tick)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_r:
                new_list = generate_starting_list(n, min_val, max_val)
                tick_count = 0
                # Reset draw panels with new list copies
                bubble_info.set_list(new_list.copy())
                insertion_info.set_list(new_list.copy())
                selection_info.set_list(new_list.copy())
                quicksort_info.set_list(new_list.copy())

                # Reinitialize generators
                bubble_sort_gen = bubble_sort(bubble_info, ascending=True)
                insertion_sort_gen = insertion_sort(insertion_info, ascending=True)
                selection_sort_gen = selection_sort(selection_info, ascending=True)
                quicksort_sort_gen = quicksort_sort(quicksort_info, ascending=True)
                bubble_tick = 0
                insertion_tick = 0
                selection_tick = 0
                quicksort_tick = 0
                # Reset step counts and timers
                bubble_steps = insertion_steps = selection_steps = quicksort_steps = 0
                bubble_elapsed_time = insertion_elapsed_time = selection_elapsed_time = quicksort_elapsed_time = 0
                bubble_sorting = insertion_sorting = selection_sorting = quicksort_sorting = True
                sorting = False

            elif event.key == pygame.K_SPACE and sorting == False:
                sorting = True
                bubble_start_time=time.time()
                insertion_start_time=time.time()
                quicksort_start_time=time.time()
                selection_start_time=time.time()

            elif event.key == pygame.K_t:
                tick_speed = get_tick_speed_input()
                window = pygame.display.set_mode((window_width, window_height))  # Restore window
                pygame.display.set_caption("Side-by-Side Sorting Algorithms")
            elif event.key == pygame.K_a:
                dataset_choice = show_dataset_selection_screen(window)
                if dataset_choice == "random":
                    starting_list = generate_starting_list(n, min_val, max_val)
                elif dataset_choice == "csv":
                    import pandas as pd
                    #filename = "sorted_lists.csv"  # or prompt for file
                    df = pd.read_csv(filename)
                    # Use the first numeric column, for example:
                    starting_list = df[df.columns[0]].dropna().astype(int).tolist()
                    csv_data = df

            elif event.key == pygame.K_n and dataset_choice == 'csv':
                print(len(csv_columns))
                if current_column <= len(csv_columns):
                    current_column += 1  # Increment the current_column to move to the next one
                elif current_column > len(csv_columns):
                    current_column = 0

                # Check if there are more columns available to display
                if df is not None and current_column < len(csv_columns):
                    # Get the next column's data
                    column_data = df.iloc[:, current_column].dropna().tolist()

                    # Truncate or pad the list to match expected length
                    if len(column_data) > n:
                        column_data = column_data[:n]
                    elif len(column_data) < n:
                        column_data += [random.randint(min_val, max_val) for _ in range(n - len(column_data))]

                    # Reinitialize sort visualizer with the new column data
                    bubble_info.set_list(column_data.copy())
                    insertion_info.set_list(column_data.copy())
                    selection_info.set_list(column_data.copy())
                    quicksort_info.set_list(column_data.copy())

                    # Reinitialize the sorting generators
                    bubble_sort_gen = bubble_sort(bubble_info, ascending=True)
                    insertion_sort_gen = insertion_sort(insertion_info, ascending=True)
                    selection_sort_gen = selection_sort(selection_info, ascending=True)
                    quicksort_sort_gen = quicksort_sort(quicksort_info, ascending=True)

                    # Reset step counts and timers
                    bubble_tick = 0
                    insertion_tick = 0
                    selection_tick = 0
                    quicksort_tick = 0
                    bubble_steps = insertion_steps = selection_steps = quicksort_steps = 0
                    bubble_elapsed_time = insertion_elapsed_time = selection_elapsed_time = quicksort_elapsed_time = 0
                    bubble_sorting = insertion_sorting = selection_sorting = quicksort_sorting = True
                    sorting = False
                else:
                    print("All CSV columns processed or CSV not loaded.")
                    print(csv_columns)
                    print(current_column)

    pygame.quit()
    export_results_to_csv(results)
    print("Results exported to results.csv.")
    if results:
        with open("sort_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

if __name__ == "__main__":
    main()