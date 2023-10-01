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
        strategy (str): The move strategy, e.g., "swap-4", "reverse-3", or "insert-2".

    Returns:
        bool: True if the move is valid, False otherwise.
    """

    # Validate strategy format
    if not (strategy.startswith("swap-") or strategy.startswith("reverse-") or strategy.startswith("insert-")):
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
        # Find the first two different elements from both ends of the after array, then reverse 
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

    elif strategy.startswith("insert-"):
        n = max_diff
        counter = 0
        largest_diff = 0
        for i in range(len(arr_before)):
            if arr_before[i] != arr_after[i]:
                diff = arr_after.index(arr_before[i]) - i  # Calculate the index difference
                # Check if the move is within the allowed range
                if abs(diff) > largest_diff:
                    largest_diff = diff
                if abs(diff) <= n and abs(diff) > 1:
                    if diff > 0:
                        for j in range(i, i + diff):
                            if arr_after[j] != arr_before[j + 1]:
                                return False  # Invalid move
                    elif diff < 0:
                        for j in range(diff + i + 1, i+1):
                            if arr_after[j] != arr_before[j - 1]:
                                return False  # Invalid move
                elif abs(diff) == 1:
                    counter += 1
                else:
                    return False  # Invalid move
        if largest_diff == 1:
            return counter == 2 # If inserts of only one-away were done, we should check that only one has been performed
        return True  # Valid move
    
# Example usage
if __name__ == '__main__':
    arr_before = [1, 2, 3, 4, 5]
    arr_after = [1, 5, 3, 4, 2]
    strategy = "swap-4"

    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4, 5]
    arr_after = [3, 2, 1, 4, 5]
    strategy = "reverse-3"

    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")
    
    arr_before = [1, 2, 3, 4 ,5]
    arr_after = [1, 3, 4, 5, 2]
    strategy = "insert-3"
    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")
