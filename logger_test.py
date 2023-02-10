import logger
log = logger.from_json('traj.json', 't')
print(log.end_time)
log2 = log.append(log).append(log).append(log).append(log).append(log).append(log)
print(log.end_time)
print(log2.end_time)

t = 20

def run():
    log2.sample(t)

print('completed one run')

import timeit
print(timeit.timeit('run()', globals=globals(), number=100000))