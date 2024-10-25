import time
import tkinter as tk
from tkinter import messagebox
import numpy as np
from numpy.linalg import det, solve
from numpy import cross, ndarray
import re
from queue import Queue
from typing import Optional

# Initialize matrix registers
registers: dict[str, Optional[np.ndarray]] = {'A': None, 'B': None, 'C': None, 'D': None, 'I': 1}
message_queue: Queue[str] = Queue()

class Matrix(np.ndarray):
    """
    Custom class that inherits from numpy.ndarray to support matrix operations
    and custom string format for copying as a matrix template.
    """

    def __str__(self) -> str:
        """
        Format matrix for output in a structured string for clipboard.

        Returns:
            str: A formatted string representation of the matrix as @MATX{{;};{;}}.
        """
        matrix_str = "@MATX{{"
        rows, cols = self.shape
        for i in range(rows):
            row_str = ";".join(map(str, self[i]))
            matrix_str += row_str
            if i != rows - 1:
                matrix_str += "};{"
        matrix_str += "}}"
        return matrix_str

def parse_matrix(elements: list[float], rows: int, cols: int) -> Matrix:
    """
    Convert a flat list of elements into a 2D numpy array for matrix form.

    Args:
        elements (list[float]): A flat list of matrix elements.
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.

    Returns:
        Matrix: A numpy Matrix object reshaped into (rows, cols).
    """
    return np.array(elements).reshape((rows, cols)).view(Matrix)

def format_element(element: str) -> str:
    """
    Format a matrix element for custom syntax interpretation of division and square root.

    Args:
        element (str): The matrix element as a string, possibly containing 'sqrt' or 'div'.

    Returns:
        str: The formatted element with syntax @DIV or @RT, if applicable.
    """
    if "sqrt" in element:
        element = re.sub(r'sqrt(\d+)', r'@RT{\1}', element)
    if "div" in element:
        parts = element.split("div")
        if len(parts) == 2:
            left = format_element(parts[0])
            right = format_element(parts[1])
            return f"@DIV{{{left};{right}}}"
    return element

def display_next_message() -> None:
    """
    Display the next message in the message queue via a popup, if any exist.
    """
    if not message_queue.empty():
        message = message_queue.get()
        messagebox.showinfo("Info", message)
        root.after(100, display_next_message)

def generate_matrix() -> None:
    """
    Generate a matrix from user input, validate elements, and store in a register if specified.
    Provides feedback messages on clipboard copy and register assignment.
    """
    try:
        rows: int = int(row_entry.get())
        cols: int = int(col_entry.get())
        matrix_elements: list[str] = matrix_entry.get().split()

        # Check for register assignment syntax "reg A", "reg B", etc.
        if len(matrix_elements) > rows * cols:
            reg_name = matrix_elements[-2].lower()
            reg_letter = matrix_elements[-1].upper()
            if reg_name != 'reg' or reg_letter not in ['A', 'B', 'C', 'D']:
                message_queue.put("Invalid register assignment. Use 'reg A', 'reg B', etc.")
                return

            matrix_elements = matrix_elements[:-2]
        else:
            reg_letter = None

        if len(matrix_elements) != rows * cols:
            message_queue.put("Number of elements doesn't match matrix size")
            return

        # Create matrix and optionally store it in a register
        matrix: Matrix = parse_matrix([float(format_element(el)) for el in matrix_elements], rows, cols)

        if reg_letter:
            registers[reg_letter] = matrix
            message_queue.put(f"Matrix stored in register {reg_letter}!")

        # Copy matrix string to clipboard
        matrix_str: str = str(matrix)
        root.clipboard_clear()
        root.clipboard_append(matrix_str)
        message_queue.put(f"Matrix copied to clipboard:\n{matrix_str}")

        # Start displaying messages
        display_next_message()

    except ValueError:
        message_queue.put("Please enter valid integers for dimensions and numbers.")

def evaluate_equation() -> None:
    """
    Evaluate the matrix equation from user input and optionally assign to a register.
    WARNING: Uses eval() which can execute arbitrary code, avoid using with untrusted input.
    """
    equation: str = equation_entry.get()
    reg_x: Optional[str] = None

    if "reg" in equation:
        equation, reg_x = equation.split("reg")
        reg_x = reg_x.upper().replace(" ", "")
        if reg_x not in registers.keys():
            message_queue.put("Invalid register assignment. Use 'reg A', 'reg B', etc.")
            reg_x = None

    try:
        # Replace register names with matrix objects
        equation = equation.replace("A", "registers['A']")
        equation = equation.replace("B", "registers['B']")
        equation = equation.replace("C", "registers['C']")
        equation = equation.replace("D", "registers['D']")
        equation = equation.replace("I", "registers['I']")

        # Eval with numpy functions and registers
        context = {'registers': registers, 'np': np, 'det': det, 'solve': solve, 'cross': cross}
        result = eval(equation, context)

        if reg_x:
            registers[reg_x] = result
            message_queue.put(f"Matrix stored in register {reg_x}!")

        message_queue.put(f"Result:\n{result}")
        display_next_message()

    except Exception as e:
        message_queue.put(f"Invalid Equation: {e}")
        display_next_message()

def call_register(lbl) -> str:
    """
    Handle selection of a register label, specifically showing a popup for identity matrix size input.

    Args:
        lbl (str): The label of the matrix to call (e.g., 'A', 'B', or 'I' for identity).

    Returns:
        str: The register label.
    """
    if lbl == 'I':
        popup = tk.Toplevel(root)
        popup.title("Choose size for Identity matrix")
        tk.Label(popup, text="Enter integer:").pack(pady=10)
        identity_entry = tk.Entry(popup)
        identity_entry.pack(pady=5, padx=12)
        identity_entry.focus_set()

        def submit_size():
            nonlocal lbl
            try:
                size_input = int(identity_entry.get())
                registers['I'] = np.eye(size_input)
                popup.destroy()
            except ValueError:
                messagebox.showerror("Invalid input", "Please enter a valid integer.")

        identity_entry.bind("<Return>", lambda e: submit_size())
        popup.wait_window()
    return lbl

def write_reg(register):
    """
    Insert text from a specified register at the current caret position in equation_entry.

    Args:
        register (str): The name of the register to insert (e.g., 'A', 'B').
    """
    if registers.get(register, None) is None:
        return
    current_position = equation_entry.index(tk.INSERT)
    equation_entry.insert(current_position, register)

def handle_shift_enter(event: tk.Event) -> None:
    """
    Handle Shift+Enter to trigger matrix generation and refocus on equation entry.
    """
    generate_matrix()
    equation_entry.focus_set()

def handle_enter(event: tk.Event) -> None:
    """
    Handle Enter key to shift focus between input fields and trigger actions.

    Args:
        event (tk.Event): Key event triggered by pressing Enter.
    """
    widget = event.widget
    if widget == row_entry:
        col_entry.focus_set()
    elif widget == col_entry:
        matrix_entry.focus_set()
    elif widget == matrix_entry:
        generate_matrix()
    elif widget == equation_entry:
        evaluate_equation()

# Initialize Tkinter window and input fields
root = tk.Tk()
root.title("Matrix Helper")

# Row and column input
tk.Label(root, text="Rows:").grid(row=0, column=0)
row_entry = tk.Entry(root)
row_entry.grid(row=0, column=1)
row_entry.focus_set()

tk.Label(root, text="Columns:").grid(row=1, column=0)
col_entry = tk.Entry(root)
col_entry.grid(row=1, column=1)

# Matrix elements input
tk.Label(root, text="Matrix Elements (space separated):").grid(row=2, column=0, columnspan=2)
matrix_entry = tk.Entry(root, width=50)
matrix_entry.grid(row=3, column=0, columnspan=2)

# Add a button to generate the matrix
generate_button = tk.Button(root, text="Generate and Copy", command=generate_matrix)
generate_button.grid(row=4, column=0, columnspan=2)

# Equation input and evaluate button
equation_entry = tk.Entry(root, width=50)
equation_entry.grid(row=5, column=0, columnspan=2)
evaluate_button = tk.Button(root, text="Evaluate Expression", command=evaluate_equation)
evaluate_button.grid(row=6, column=1, columnspan=1)

# Use enumerate to create buttons and place them in the grid
button_frame = tk.Frame(root)

for index, label in enumerate(("A", "B", "C", "D", "I")):
    button = tk.Button(button_frame, text=label, command=lambda lbl=label: write_reg(call_register(lbl)))
    button.grid(row=0, column=index, padx=5, pady=5)

button_frame.grid(row=6, column=0, columnspan=1)

# Bind events for Enter key handling
root.bind('<Return>', handle_enter)
root.bind('<Shift-Return>', handle_shift_enter)

root.mainloop()
