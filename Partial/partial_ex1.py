import random

jucatorStart = random.randint(0,1)
print(jucatorStart)

castiguriJucator1 = 0
castiguriJucator2 = 0
for i in range(20000):
  n = 0
  m = 0

  if(jucatorStart == 0):
    if random.uniform(0,1) <= 1/3:
      n = 1
  else:
    if random.randint(0,1) == 1:
      n = 1

  for i in range(n+1):
    if jucatorStart == 0:
      if random.randInt(0,1) == 1:
        m += 1
    else:
      if random.uniform(0,1) <=1/3:
        m += 1

  if n >= m:
    castiguriJucator1 += 1
  else:
    castiguriJucator2 += 1

print(castiguriJucator1) #aproximativ 15000-15500 dupa mai multe rulari
print(castiguriJucator2) #aproximativ 4500-5000 dupa mai multe rulari

#P0 are sanse mai mari de castig
