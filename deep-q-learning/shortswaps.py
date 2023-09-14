import numpy as np
import subprocess as sub

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

class ShortSwap():
  def __init__(self, n):
    self.state = np.array([i+1 for i in range(n)])
    self.observation_space_shape = (n,)
    self.observation_space_size = np.math.factorial(n)
    self.action_space_size = 2*n-3
    self.num_moves = 0
    self.optimal_moves = 0
    self.visited = set()

  def get_permutation_from_index(self, k):
    n = self.observation_space_shape[0]
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

    return p
  
  def get_index_from_permutation(self, state):
    n = self.observation_space_shape[0]
    ordered_set = OrderedSet(n)
    for i in range(n):
      ordered_set.insert(i+1)

    index = 0
    for i in range(n):
      index += ordered_set.order_of_key(state[i])*np.math.factorial(n-i-1)
      ordered_set.erase(state[i])
    return index

  def reset(self):
    self.state = self.get_permutation_from_index(np.random.randint(1, self.observation_space_size))
    self.num_moves = 0
    self.visited = set()

    sub_result = sub.run(["../solver.exe", str(self.observation_space_shape[0]), *[str(x) for x in self.state]], stdout=sub.PIPE)
    self.optimal_moves = int(sub_result.stdout.decode())

    self.visited.add(self.get_index_from_permutation(self.state))
    return self.state
  
  def get_valid_moves(self):
    n = self.observation_space_shape[0]
    valid_moves = [1]*self.action_space_size
    for action in range(self.action_space_size):
      move = action//(n-1)+1
      pos = action%(n-1)
      self.state[pos], self.state[pos+move] = self.state[pos+move], self.state[pos]
      index = self.get_index_from_permutation(self.state)

      if index in self.visited:
        valid_moves[action] = 0
      self.state[pos], self.state[pos+move] = self.state[pos+move], self.state[pos]

    return np.array([i for i in range(self.action_space_size) if valid_moves[i] == 1])

  def action_space_sample(self):
    valid_moves = self.get_valid_moves()
    return np.random.choice(valid_moves, size=1)[0]
  
  def step(self, action):
    n = self.observation_space_shape[0]
    move = action//(n-1)+1
    pos = action%(n-1)
    self.state[pos], self.state[pos+move] = self.state[pos+move], self.state[pos]
    self.num_moves += 1
    self.visited.add(self.get_index_from_permutation(self.state))

    if all(self.state == np.array(range(1, n+1))):
      reward = 100**(1/((self.num_moves-self.optimal_moves)+1))
      done = True
    elif len(self.get_valid_moves()) == 0:
      reward = 0.0
      done = True
    else:
      reward = 0.0
      done = False

    return self.state, reward, done

  def render(self):
    print(self.num_moves, self.state)

