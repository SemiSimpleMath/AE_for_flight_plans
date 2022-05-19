import torch
import random


def create_flight_plan_dict(s):
    d = {}

    for i, c in enumerate(s):
        d[c] = i

    return d


def convert_string_flight_plan_to_torch(input):
    s = "NEWSC0123456789e"
    d = create_flight_plan_dict(s)
    MAX_PLAN_LENGTH = 16
    CHAR_ENCODE_LENGTH = 16

    torch_encoded = []

    for i in range(MAX_PLAN_LENGTH):
        result = torch.zeros(CHAR_ENCODE_LENGTH)
        if i < len(input):
            c = input[i]
            loc = d[c]

        else:
            loc = d['0']
        result[loc] = 1
        result = (result - torch.mean(result)) / torch.std(result)
        torch_encoded.append(result)
    final = torch.stack(torch_encoded, 0)
    return final


# n[int]: the number of plans to generate
# d the dictionary used to encode the flight plan
# file[string]: the file to write the plans into
def generate_flight_plans(n, d):
    MAX_PLAN_LENGTH = 16
    plans = []
    for i in range(n):
        # generate the flight plan length [1, 16]

        plan_length = random.randint(1, MAX_PLAN_LENGTH)
        plan = ""

        # first char is always a letter

        r = random.randint(0, 4)
        c = d[r]
        plan += c

        for i in range(plan_length - 2):
            r = random.randint(0, len(d) - 1)
            plan += d[r]

        plan += 'e'
        plans.append(plan)
    return plans


def write_plans(file_name, plans):
    import csv
    with open(file_name, 'w') as file:
        writer = csv.writer(file)
        header = ["input", "target"]
        writer.writerow(header)
        for line in plans:
            writer.writerow([line, line])


