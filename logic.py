import re
from typing import Optional
import numpy as np
from numpy.linalg import det, solve
from numpy import cross

# Matrix registers shared across UI and logic
registers: dict[str, Optional[np.ndarray]] = {'A': None, 'B': None, 'C': None}

class Matrix(np.ndarray):
    """numpy ndarray subclass with custom string representation."""

    def __str__(self) -> str:
        matrix_str = "@MATX{"
        rows, cols = self.shape
        for i in range(rows):
            row_str = ";".join(map(str, self[i]))
            matrix_str += row_str
            if i != rows - 1:
                matrix_str += "};{"
        matrix_str += "}"
        return matrix_str


def format_element(element: str) -> str:
    """Format matrix element for @DIV and @RT syntax."""
    if "sqrt" in element:
        element = re.sub(r'sqrt(\d+)', r'@RT{\1}', element)
    if "div" in element:
        parts = element.split("div")
        if len(parts) == 2:
            left = format_element(parts[0])
            right = format_element(parts[1])
            return f"@DIV{{{left};{right}}}"
    return element


def parse_matrix(elements: list[float], rows: int, cols: int) -> Matrix:
    """Convert list of floats into a Matrix object."""
    return np.array(elements).reshape((rows, cols)).view(Matrix)


def create_matrix(elements: list[str], rows: int, cols: int, reg_letter: Optional[str] = None) -> Matrix:
    """Create a matrix and optionally store it in a register."""
    formatted = [float(format_element(el)) for el in elements]
    matrix = parse_matrix(formatted, rows, cols)
    if reg_letter:
        if reg_letter not in registers:
            raise ValueError("Invalid register letter")
        registers[reg_letter] = matrix
    return matrix


def evaluate_expression(expression: str) -> tuple[np.ndarray, Optional[str]]:
    """Evaluate an expression using numpy and the matrix registers."""
    reg_x: Optional[str] = None
    if "reg" in expression:
        expression, reg_x = expression.split("reg")
        reg_x = reg_x.upper().replace(" ", "")
        if reg_x not in registers:
            raise ValueError("Invalid register assignment")

    expression = expression.replace("A", "registers['A']")
    expression = expression.replace("B", "registers['B']")
    expression = expression.replace("C", "registers['C']")

    context = {
        'registers': registers,
        'np': np,
        'det': det,
        'solve': solve,
        'cross': cross
    }
    result = eval(expression, context)
    if reg_x:
        registers[reg_x] = result
    return result, reg_x
