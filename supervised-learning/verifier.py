import numpy as np

def are_elements_in_array(arr, i, j):
    """
    Check if a sorting move is valid.
    
    Parameters:
    - arr: The array or list to be sorted.
    - i: Index of the first element to be swapped.
    - j: Index of the second element to be swapped.
    
    Returns:
    - True if the elements are in the array
    - False otherwise.
    """
    # Check if indices are within the bounds of the array.
    if 0 <= i < len(arr) and 0 <= j < len(arr):
        return True
    else:
        # If indices are out of bounds, the move is invalid.
        return False

def is_valid_array(arr):
    """
    Check if an array is valid.

    An array is considered valid if it meets the following criteria:
    1. It contains no duplicate elements.
    2. All elements are within the range of 1 to n, where n is the length of the array.

    Parameters:
    arr (list): The input array to be checked.

    Returns:
    bool: True if the array is valid, False otherwise.
    
    Example:
    >>> is_valid_array([1, 2, 3, 4, 5])
    True
    >>> is_valid_array([1, 2, 2, 4, 5])
    False
    >>> is_valid_array([1, 2, 3, 4, 6])
    False
    """
    n = len(arr)
    seen = set()
    
    for num in arr:
        if num in seen or num < 1 or num > n:
            return False
        seen.add(num)
    
    return True
