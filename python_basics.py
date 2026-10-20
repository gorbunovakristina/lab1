"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    count = 0
    for t in text.lower():
        if t in 'aeiou':
            count += 1
    return count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    count_letters = {}
    for t in text:
        if t not in count_letters:
            count_letters[t] = 1
        else:
            count_letters[t] += 1

    flag = True
    for letter in count_letters:
        if count_letters[letter] > 1:
            flag = False
            break

    return flag


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = bin(data.value)
    count = 0
    for n in str(number):
        if n == '1':
            count += 1
    return count



def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while len(str(number)) > 1:
        multiplication = 1
        for n in str(number):
            multiplication *= int(n)
        number = multiplication
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    summ = 0
    for i in range(n):
        summ += (predicted[i] - expected[i]) ** 2
    return summ / n


def is_prime(number) -> bool:
    if number <= 1:
        return False

    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False

    return True

def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    power_number = {}
    s = ''
    i = 2
    while number > 1:
        if number % i == 0 and is_prime(i):
            if i not in power_number:
                power_number[i] = 1
            else:
                power_number[i] += 1
            number //= i
        else:
            i += 1

    for k in power_number:
        if power_number[k] == 1:
            s += f'({k})'
        else:
            s += f'({k}**{power_number[k]})'
    return s


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    i = 0
    while cube_count > 0:
        i += 1
        cube_count -= i ** 2
    if cube_count == 0:
        return i

    else:
        return 'It is impossible'


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    str_number = str(number)
    sum_1 = 0
    sum_2 = 0

    if len(str_number) % 2 == 0:
        for i in range(len(str_number) // 2 - 1):
            sum_1 += int(str_number[i])
            sum_2 += int(str_number[-i - 1])

    else:
        for i in range(len(str_number) // 2):
            sum_1 += int(str_number[i])
            sum_2 += int(str_number[-i - 1])

    if sum_1 == sum_2:
        return True
    else:
        return False
