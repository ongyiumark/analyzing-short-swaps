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

def check_swap(arr_before, arr_after, max_diff):
    swap_count = 0
    # Iterate through both arrays and compare the elements
    for i in range(len(arr_before)):
        # Calculate the absolute difference between the positions of the same element
        pos_diff = abs(arr_before.index(arr_after[i]) - i)

        # Check if the absolute difference is within the allowed range
        if pos_diff >= max_diff:
            return False

        if arr_before[i] != arr_after[i]:
            swap_count += 1
        
    return swap_count == 2 #Only two elements should be in different places

def check_reversal(arr_before, arr_after, max_diff):
        # Find the first two different elements from both ends of the after array, then reverse 
        first_diff_idx = None
        last_diff_idx = None
        for i in range(len(arr_before)):
            if arr_before[i] != arr_after[i]:
                if first_diff_idx is None:
                    first_diff_idx = i
                last_diff_idx = i
        if first_diff_idx is None or first_diff_idx == last_diff_idx or last_diff_idx - first_diff_idx >= max_diff:
            return False
        arr_after_copy = arr_after[:]
        arr_after_copy[first_diff_idx:last_diff_idx + 1] = arr_after[first_diff_idx:last_diff_idx + 1][::-1]

        return arr_after_copy == arr_before

def check_insert(arr_before, arr_after, max_diff):
    after_diffs = []
    ups = 0
    downs = 0
    big_diff = 0
    for pos in range(len(arr_before)):
        element = arr_before[pos]
        after_pos = arr_after.index(element)
        diff = after_pos - pos
        big_diff = max(big_diff, abs(diff))
        if diff > 0: ups += 1
        elif diff < 0: downs += 1
        after_diffs.append(diff)
    
    if (big_diff >= max_diff) or big_diff == 0:
        return False
    if (ups > 1 and downs > 1):
        return False
    
    if ups < downs: comp = max(after_diffs)
    else: comp = min(after_diffs)

    if comp >= 1: return comp==after_diffs.count(-1)
    else:  return abs(comp)==after_diffs.count(1)

def check_block(arr_before, arr_after, max_diff):
    first_idx = None
    last_idx = None

    for pos in range(len(arr_before)):
        if arr_before[pos] != arr_after[pos]:
            first_idx = pos
            break
    for pos in range(len(arr_before)-1, -1, -1):
        if arr_before[pos] != arr_after[pos]:
            last_idx = pos
            break
    
    if first_idx is None or last_idx is None or (last_idx - first_idx == 0) or (last_idx - first_idx >= max_diff):
        return False
    
    first_diff = arr_after.index(arr_before[first_idx]) - first_idx
    anchor_idx = first_idx + 1
    for pos in range(first_idx, last_idx):
        diff = arr_after.index(arr_before[pos]) - pos
        if diff != first_diff:
            anchor_idx = pos
            break
    
    comp_arr = arr_before[0:first_idx] + arr_before[anchor_idx:last_idx+1] + arr_before[first_idx:anchor_idx] + arr_before[last_idx+1:]
    return arr_after == comp_arr


def is_valid_move(arr_before, arr_after, strategy):
    """
    Check if a move is valid based on the given strategy.

    Parameters:
        arr_before (list): The initial array before the move.
        arr_after (list): The resulting array after the move.
        strategy (str): The move strategy, e.g., "swap-4", "reverse-3", "insert-2", or "block-n".

    Returns:
        bool: True if the move is valid, False otherwise.
    """

    # Validate strategy format
    if not (strategy.startswith("swap-") or strategy.startswith("reverse-") or strategy.startswith("insert-") or strategy.startswith("block-")):
        raise ValueError("Invalid strategy format. Must start with 'swap-', 'reverse-', 'insert-' or 'block-'.")

    try:
        # Extract the maximum allowed difference from the strategy
        if strategy.split('-')[1] == 'n':
            max_diff = float('inf')
        else:
            max_diff = int(strategy.split('-')[1])
    except ValueError:
        raise ValueError("Invalid strategy format. Must be 'x-n' where x is the strategy name, n is an integer or the string n.")

    # Check if the lengths of both arrays are the same
    if len(arr_before) != len(arr_after):
        return False

    # Check the strategy
    if strategy.startswith("swap-"):
        return check_swap(arr_before, arr_after, max_diff)
    
    elif strategy.startswith("reverse-"):
        return check_reversal(arr_before, arr_after, max_diff)

    elif strategy.startswith("insert-"):
        return check_insert(arr_before, arr_after, max_diff)        
    
    elif strategy.startswith("block-"):
        return check_block(arr_before, arr_after, max_diff)
   

    
# Example usage
if __name__ == '__main__':
    arr_before = [1, 2, 3, 4]
    arr_after = [4, 2, 3, 1]
    strategy = "swap-3"

    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4]
    arr_after = [4, 3, 2, 1]
    strategy = "reverse-3"

    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")
    
    arr_before = [1, 2, 3, 4]
    arr_after = [2, 3, 4, 1]
    strategy = "insert-3"
    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4, 5, 6, 7, 8]
    arr_after = [1, 4, 5, 6, 7, 2, 3, 8]
    strategy = "block-6"
    if is_valid_move(arr_before, arr_after, strategy):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    print("-----INSERTS------")
    arr_before = [1, 2, 3, 4]
    arr_after = [2, 3, 4, 1]
    if check_insert(arr_before, arr_after, 3):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4, 5, 6, 7, 8]
    arr_after = [1, 3, 4, 2, 5, 7, 8, 6]
    if is_valid_move(arr_before, arr_after, "insert-3"):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4]
    arr_after = [2, 3, 4, 1]
    if check_insert(arr_before, arr_after, 4):
        print("The move is valid.")
    else:
        print("The move is not valid.")
    

    print("-----BLOCK MOVES------")
    arr_before = [1, 2, 3, 4, 5]
    arr_after = [3, 4, 5, 1, 2]
    if check_block(arr_before, arr_after, 5):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4, 5]
    arr_after = [3, 4, 1, 2, 5]
    if check_block(arr_before, arr_after, 4):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4, 5]
    arr_after = [1, 2, 4, 3, 5]
    if check_block(arr_before, arr_after, 5):
        print("The move is valid.")
    else:
        print("The move is not valid.")

    arr_before = [1, 2, 3, 4, 5]
    arr_after = [1, 3, 2, 5, 4]
    if is_valid_move(arr_before, arr_after, "block-5"):
        print("The move is valid.")
    else:
        print("The move is not valid.")