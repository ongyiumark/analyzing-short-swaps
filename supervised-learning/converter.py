import numpy as np

class OrderedSet():
  def __init__(self, n):
    self.size = 1
    while self.size < n: 
      self.size <<= 1

    self.BIT = [0]*(self.size+1)

  def insert(self, x):
    while x <= self.size:
      self.BIT[x] += 1
      x += (x&-x)
  
  def erase(self, x):
    while x <= self.size:
      self.BIT[x] -= 1
      x += (x&-x)
    
  def order_of_key(self, x):
    result = 0
    while x > 0:
      result += self.BIT[x]
      x -= (x&-x)
    
    return result-1
  
  def find_by_order(self, k):
    x = self.size
    total = 0
    result = 0

    while x > 0:
      if total + self.BIT[result+x] <= k:
        total += self.BIT[result+x]
        result += x
      x >>= 1
    
    return result+1

def get_permutation_from_index(k, n):
  """
  Returns the size-n permuation of a given index

  Parameters:
    k (int): The index of the permutation (0 <= k < n!)
    n (int): The size of the permutation
  
  Returns:
    numpy.ndarray[int]: The resulting permutation
  """
  p = [0]*n
  ordered_set = OrderedSet(n)
  for i in range(n):
    ordered_set.insert(i+1)

  for i in range(n):
    lo = 0
    hi = n-i-1
    ans = 0

    while lo <= hi:
      mid = hi - (hi-lo)//2
      if mid*np.math.factorial(n-i-1) <= k:
        ans = mid
        lo = mid+1
      else:
        hi = mid-1

    p[i] = ordered_set.find_by_order(ans)
    k -= ans*np.math.factorial(n-i-1)
    ordered_set.erase(p[i])

  p = np.array(p, dtype=np.int32)
  return p
  
def get_index_of_permutation(state):
  """
  Returns the index of a permutation

  Parameters:
    state (numpy.ndarray[int]): The permutation
  
  Returns:
    int: The index of the permutaton
  """
  n = len(state)
  ordered_set = OrderedSet(n)
  for i in range(n):
    ordered_set.insert(i+1)

  index = 0
  for i in range(n):
    index += ordered_set.order_of_key(state[i])*np.math.factorial(n-i-1)
    ordered_set.erase(state[i])
  return index