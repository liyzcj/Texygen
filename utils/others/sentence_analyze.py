import os
from collections import Counter



def count_gram(file, gram):
  with open(file) as f:
    c = f.readlines()

  c = [l.strip().split() for l in c]
  split = [line[:gram] for line in c]
  split = [' '.join(s) for s in split]

  counter = Counter(split)
  return counter

if __name__ == "__main__":

  data_loc = os.path.join('data', 'image_coco.txt')
  count = []
  for gram in range(1,11):
    a = count_gram(data_loc, gram)
    count.append(len(a))
    print(f"Gram:{gram}, Length: {len(a)}")

  import matplotlib.pyplot as plt
  plt.plot(count)
  plt.show()