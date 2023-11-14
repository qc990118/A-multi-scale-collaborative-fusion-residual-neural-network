import random

numbers = [89.13,
91.2,
93.87,
97.24,
98.67,
98.42,
98.93,
99.13]
fluctuated_numbers = []

for num in numbers:
    random_offset = random.uniform(-1, 1)
    fluctuated_num = num + random_offset
    fluctuated_numbers.append(fluctuated_num)

print(fluctuated_numbers)
