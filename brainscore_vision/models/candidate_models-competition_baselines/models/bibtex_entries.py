import csv
import os

entries = {}


def find_entry(model):
    if len(entries) == 0:
        with open(f'{os.path.dirname(__file__)}/../candidate_models/base_models/models.csv') as refs:
            csv_reader = csv.reader(refs, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    entries[row[0]] = row[2]
                line_count += 1

    if model in entries:
        return entries[model]
    return ''
