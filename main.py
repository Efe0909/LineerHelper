import tkinter as tk
from tkinter import messagebox
import numpy as np
from numpy.linalg import det, solve
from numpy import cross
import re
from queue import Queue
from typing import Optional

# Initialize matrix registers
registers: dict[str, Optional[np.ndarray]] = {'A': None, 'B': None, 'C': None}
message_queue: Queue[str] = Queue()

class Matrix(np.ndarray):
    """Custom class that inherits from numpy.ndarray to support matrix operations and custom __str__ format."""

    def __str__(self) -> str:
        """
        Format matrix as @MATX{{;};{;}} for matrix output.

        Returns:
            str: The formatted string representation of the matrix.
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
    """Convert the list of elements into a numpy array.

    Args:
        elements (list[float]): The elements of the matrix.
        rows (int): The number of rows.
        cols (int): The number of columns.

    Returns:
        Matrix: A matrix object created from the elements.
    """
    return np.array(elements).reshape((rows, cols)).view(Matrix)

def format_element(element: str) -> str:
    """Format matrix element for @DIV and @RT syntax.

    Args:
        element (str): The matrix element to format.

    Returns:
        str: The formatted element.
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
    """Display the next message in the queue, if available."""
    if not message_queue.empty():
        message = message_queue.get()
        messagebox.showinfo("Info", message)
        root.after(100, display_next_message)

def generate_matrix() -> None:
    """Generate a matrix from user input and optionally store it in a register."""
    try:
        # Get rows and columns from input
        rows: int = int(row_entry.get())
        cols: int = int(col_entry.get())
        # Get matrix elements split by spaces
        matrix_elements: list[str] = matrix_entry.get().split()

        # Check for "reg A", "reg B", etc. at the end
        if len(matrix_elements) > rows * cols:
            reg_name = matrix_elements[-2].lower()  # second last token should be 'reg'
            reg_letter = matrix_elements[-1].upper()  # last token should be A, B, or C
            if reg_name != 'reg' or reg_letter not in ['A', 'B', 'C']:
                message_queue.put("Invalid register assignment. Use 'reg A', 'reg B', or 'reg C'.")
                return
            matrix_elements = matrix_elements[:-2]  # Remove 'reg X' from the elements
        else:
            reg_letter = None  # No register provided

        # Check if the number of elements matches rows * cols
        if len(matrix_elements) != rows * cols:
            message_queue.put("Number of elements doesn't match matrix size")
            return

        # Create the matrix and optionally store it in a register
        matrix: Matrix = parse_matrix([float(format_element(el)) for el in matrix_elements], rows, cols)

        if reg_letter:
            registers[reg_letter] = matrix
            message_queue.put(f"Matrix stored in register {reg_letter}!")

        # Copy matrix string to clipboard
        matrix_str: str = str(matrix)
        root.clipboard_clear()  # Clear previous clipboard contents
        root.clipboard_append(matrix_str)  # Copy the matrix string
        message_queue.put(f"Matrix copied to clipboard:\n{matrix_str}")

        # Start displaying messages
        display_next_message()

    except ValueError:
        message_queue.put("Please enter valid integers for dimensions and numbers.")

def open_equation_window() -> None:
    """Open the equation entry window for matrix operations."""
    eq_window = tk.Toplevel(root)
    eq_window.title("Matrix Operations")

    tk.Label(eq_window, text="Enter Equation:").pack()
    equation_entry = tk.Entry(eq_window, width=50)
    equation_entry.pack()

    equation_entry.focus_set()
    def evaluate_equation() -> None:
        """Evaluate the matrix equation entered by the user.
        !!! It's not a secure implementation as it uses eval(). """
        equation: str = equation_entry.get()
        reg_x: Optional[str] = None

        # Check for register assignment
        if "reg" in equation:
            equation, reg_x = equation.split("reg")
            reg_x = reg_x.upper().replace(" ", "")
            if reg_x not in registers.keys():
                message_queue.put("Invalid register assignment. Use 'reg A', 'reg B', or 'reg C'.")
                reg_x = None

        try:
            # Replace register names with the matrix objects
            equation = equation.replace("A", "registers['A']")
            equation = equation.replace("B", "registers['B']")
            equation = equation.replace("C", "registers['C']")

            # Add linear algebra functions to the eval context
            context = {
                'registers': registers,
                'np': np,
                'det': det,
                'solve': solve,
                'cross': cross
            }

            # Evaluate the expression using numpy functions
            result = eval(equation, context)

            if reg_x:
                registers[reg_x] = result
                message_queue.put(f"Matrix stored in register {reg_x}!")

            message_queue.put(f"Result:\n{result}")
            display_next_message()

        except Exception as e:
            message_queue.put(f"Invalid Equation: {e}")
            display_next_message()

    eq_window.bind('<Escape>', lambda e: eq_window.destroy())

    evaluate_button = tk.Button(eq_window, text="Evaluate", command=evaluate_equation)
    evaluate_button.pack()

# Function to handle Shift+Enter key press
def handle_shift_enter(event: tk.Event) -> None:
    """Handle Shift+Enter key press to generate a matrix."""
    generate_matrix()

# Function to handle 'O' key press for opening equation window
def handle_o_key(event: tk.Event) -> None:
    """Handle 'O' key press to open the equation window."""
    open_equation_window()

# Function to handle Enter key press for focus changing
def handle_enter(event: tk.Event) -> None:
    """Handle Enter key press to change focus between input fields."""
    widget = event.widget
    if widget == row_entry:
        col_entry.focus_set()  # Move focus to columns entry
    elif widget == col_entry:
        matrix_entry.focus_set()  # Move focus to matrix entry
    elif widget == matrix_entry:
        generate_matrix()  # Generate matrix if matrix entry is focused

# Initialize the Tkinter window
root = tk.Tk()
root.title("Matrix Input Helper")

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

# Button to open equation window
equation_button = tk.Button(root, text="Open Equation Window", command=open_equation_window)
equation_button.grid(row=5, column=0, columnspan=2)

# Bind Shift+Enter to generate matrix and copy
root.bind('<Shift-Return>', handle_shift_enter)

# Bind 'O' key to open equation window
root.bind('<o>', handle_o_key)

# Bind Enter to change focus
root.bind('<Return>', handle_enter)

# Run the Tkinter event loop (until the user exits)
root.mainloop()
