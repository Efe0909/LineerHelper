import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue
from typing import Optional

from logic import create_matrix, evaluate_expression, registers

# Queue for sequential message popups
message_queue: Queue[str] = Queue()


def display_next_message() -> None:
    """Display the next message in the queue, if available."""
    if not message_queue.empty():
        message = message_queue.get()
        messagebox.showinfo("Info", message)
        root.after(100, display_next_message)


def generate_matrix() -> None:
    """Generate matrix from user input and handle clipboard/storage."""
    try:
        rows = int(row_var.get())
        cols = int(col_var.get())
        matrix_elements = matrix_var.get().split()

        if len(matrix_elements) > rows * cols:
            reg_name = matrix_elements[-2].lower()
            reg_letter = matrix_elements[-1].upper()
            if reg_name != 'reg' or reg_letter not in registers:
                message_queue.put("Invalid register assignment. Use 'reg A', 'reg B', or 'reg C'.")
                display_next_message()
                return
            matrix_elements = matrix_elements[:-2]
        else:
            reg_letter = None

        if len(matrix_elements) != rows * cols:
            message_queue.put("Number of elements doesn't match matrix size")
            display_next_message()
            return

        matrix = create_matrix(matrix_elements, rows, cols, reg_letter)
        if reg_letter:
            message_queue.put(f"Matrix stored in register {reg_letter}!")

        root.clipboard_clear()
        root.clipboard_append(str(matrix))
        message_queue.put(f"Matrix copied to clipboard:\n{matrix}")
        display_next_message()
    except ValueError:
        message_queue.put("Please enter valid integers for dimensions and numbers.")
        display_next_message()


def open_equation_window() -> None:
    """Open a window to evaluate matrix expressions."""
    eq_window = tk.Toplevel(root)
    eq_window.title("Matrix Operations")

    ttk.Label(eq_window, text="Enter Equation:").pack(pady=5)
    equation_entry = ttk.Entry(eq_window, width=50)
    equation_entry.pack(pady=5)
    equation_entry.focus_set()

    def evaluate_equation() -> None:
        equation = equation_entry.get()
        try:
            result, reg_x = evaluate_expression(equation)
            if reg_x:
                message_queue.put(f"Matrix stored in register {reg_x}!")
            message_queue.put(f"Result:\n{result}")
        except Exception as e:  # noqa: BLE001
            message_queue.put(f"Invalid Equation: {e}")
        display_next_message()

    eq_window.bind('<Escape>', lambda e: eq_window.destroy())
    ttk.Button(eq_window, text="Evaluate", command=evaluate_equation).pack(pady=5)


def handle_shift_enter(event: tk.Event) -> None:
    """Generate matrix on Shift+Enter."""
    generate_matrix()


def handle_o_key(event: tk.Event) -> None:
    """Open equation window on 'O' key press."""
    open_equation_window()


def handle_enter(event: tk.Event) -> None:
    """Change focus between input fields on Enter."""
    widget = event.widget
    if widget == row_entry:
        col_entry.focus_set()
    elif widget == col_entry:
        matrix_entry.focus_set()
    elif widget == matrix_entry:
        generate_matrix()


root = tk.Tk()
root.title("Matrix Input Helper")
root.resizable(False, False)

style = ttk.Style(root)
try:
    style.theme_use('clam')
except Exception:
    pass

main_frame = ttk.Frame(root, padding=10)
main_frame.grid(row=0, column=0)

row_var = tk.StringVar()
col_var = tk.StringVar()
matrix_var = tk.StringVar()

# Row and column inputs
ttk.Label(main_frame, text="Rows:").grid(row=0, column=0, sticky="e")
row_entry = ttk.Entry(main_frame, textvariable=row_var, width=5)
row_entry.grid(row=0, column=1, sticky="w")
row_entry.focus_set()

ttk.Label(main_frame, text="Columns:").grid(row=1, column=0, sticky="e")
col_entry = ttk.Entry(main_frame, textvariable=col_var, width=5)
col_entry.grid(row=1, column=1, sticky="w")

# Matrix elements input
ttk.Label(main_frame, text="Matrix Elements (space separated):").grid(row=2, column=0, columnspan=2, pady=(10, 0))
matrix_entry = ttk.Entry(main_frame, textvariable=matrix_var, width=40)
matrix_entry.grid(row=3, column=0, columnspan=2, pady=(0, 10))

# Buttons
generate_button = ttk.Button(main_frame, text="Generate and Copy", command=generate_matrix)
generate_button.grid(row=4, column=0, columnspan=2, pady=5)

equation_button = ttk.Button(main_frame, text="Open Equation Window", command=open_equation_window)
equation_button.grid(row=5, column=0, columnspan=2)

# Bindings
root.bind('<Shift-Return>', handle_shift_enter)
root.bind('<o>', handle_o_key)
root.bind('<Return>', handle_enter)

root.mainloop()
