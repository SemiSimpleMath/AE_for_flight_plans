import torch
import random

def create_flight_plan_dict(s):


    d = {}

    for i, c in enumerate(s):
        d[c] = i

    return d

def create_char_list(s):
    return s.split()

def convert_string_flight_plan_to_torch(input):
    s = "NEWSC0123456789e"
    d = create_flight_plan_dict(s)
    MAX_PLAN_LENGTH = 16
    CHAR_ENCODE_LENGTH = 16

    result = torch.zeros(MAX_PLAN_LENGTH*CHAR_ENCODE_LENGTH)



    for i, c in enumerate(input):
        loc = d[c] + CHAR_ENCODE_LENGTH * i
        result[loc] = 1

    return result



# n[int]: the number of plans to generate
# d the dictionary used to encode the flight plan
# file[string]: the file to write the plans into
def generate_flight_plans(n,d):
    MAX_PLAN_LENGTH = 16
    plans = []
    for i in range(n):
        # generate the flight plan length [1, 16]

        plan_length = random.randint(1, MAX_PLAN_LENGTH)
        plan = ""

        # first char is always a letter

        r = random.randint(0,4)
        c = d[r]
        plan += c

        for i in range(plan_length-2):

            r = random.randint(0, len(d)-1)
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
            writer.writerow([line,line])



"""

input = 'N7S3E2e'

convert_string_flight_plan_to_torch(d, input)


s = "NEWSC0123456789e"
d = list(s)
plans = generate_flight_plans(1, d)

write_plans("./data/flight_plans/plans1.csv", plans)
"""