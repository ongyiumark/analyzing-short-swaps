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


def is_valid_move(arr_before, arr_after, strategy):
    """
    Check if a move is valid based on the given strategy.

    Parameters:
        arr_before (list): The initial array before the move.
        arr_after (list): The resulting array after the move.
        strategy (str): The move strategy, e.g., "swap-4" or "reverse-3".

    Returns:
        bool: True if the move is valid, False otherwise.
    """

    # Validate strategy format
    if not (strategy.startswith("swap-") or strategy.startswith("reverse-")):
        raise ValueError("Invalid strategy format. Must start with 'swap-' or 'reverse-'.")

    try:
        # Extract the maximum allowed difference from the strategy
        if strategy.split('-')[1] == 'n':
            max_diff = float('inf')
        else:
            max_diff = int(strategy.split('-')[1])
    except ValueError:
        raise ValueError("Invalid strategy format. Must be 'swap-n' or 'reverse-n' where n is an integer or the string n.")

    # Check if the lengths of both arrays are the same
    if len(arr_before) != len(arr_after):
        return False

    # Define a function to reverse a portion of the array
    def reverse_subarray(arr, i, j):
        return arr[:i] + arr[i:j+1][::-1] + arr[j+1:]

    # Check if the strategy is a swap or reverse operation
    if strategy.startswith("swap-"):
        swap_count = 0 
        # Iterate through both arrays and compare the elements
        for i in range(len(arr_before)):
            # Calculate the absolute difference between the positions of the same element
            pos_diff = abs(arr_before.index(arr_after[i]) - i)

            # Check if the absolute difference is within the allowed range
            if pos_diff > max_diff:
                return False

            if arr_before[i] != arr_after[i]:
                swap_count += 1
            
        return swap_count == 2 #Only two elements should be in different places
    
    elif strategy.startswith("reverse-"):
        # Find the first two different elements from both ends of the after array, then
        first_diff_idx = None
        last_diff_idx = None
        for i in range(len(arr_before)):
            if arr_before[i] != arr_after[i]:
                if first_diff_idx is None:
                    first_diff_idx = i
                last_diff_idx = i
        if first_diff_idx is None or first_diff_idx == last_diff_idx or last_diff_idx - first_diff_idx > max_diff:
            return False
        arr_after_copy = arr_after[:]
        arr_after_copy[first_diff_idx:last_diff_idx + 1] = arr_after[first_diff_idx:last_diff_idx + 1][::-1]

        return arr_after_copy == arr_before
    # If all checks pass, the move is valid
    return True

# Example usage:
if __name__ == '__main__':
    arr_before = [1, 2, 3, 4, 5]
    arr_after = [1, 5, 3, 4, 2]
    strategy = "swap-4"

    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4, 5]
    arr_after = [3, 2, 1, 5, 4]
    strategy = "reverse-3"

    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")
