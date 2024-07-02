import pandas as pd
import re

human_data = pd.read_csv('human_data_normal/human_data.csv')
images_shown = pd.read_csv('human_data_normal/images_shown.csv')
human_data["image_shown"] = "FILL"

columns = {"Participants 1-7": [1,2,3,4,5,6,7],"Participants 8-13": [8,9,10,11,12,13],
           "Participants 14-19": [14,15,16,17,18,19],"Participants 20-24": [20,21,22,23,24],
           "Participants 25-28": [25,26,27,28,29], "Participants 29-32": [29,30,31,32]}
categories = ['bear', 'bunny', 'cat', 'elephant', 'frog', 'lizard', 'tiger', 'turtle', 'wolf']
mapping = {"o": "fragmented", "w": "whole", "f": "Frankenstein"}

def get_parts(image_id):
    match = re.match(r"([a-z]+)([0-9]+)", image_id, re.I)
    if match:
        items = match.groups()
    else:
        items = ["", ""]
    # ground truth
    ground_truth = items[0]
    return ground_truth


for i in range(0, len(human_data)):
    subject_number = human_data["Subj"][i]
    column_to_look_at = ""
    for header in columns:
        if subject_number in columns[header]:
            column_to_look_at = header
            break
    condition = human_data["Frankensteinonfig"][i]
    animal = human_data["Animal"][i]

    # clear seen every subject
    if i % 32 == 0:
        seen = []

    for image_name in images_shown[column_to_look_at]:

        ground_truth = get_parts(image_name)
        if ground_truth in categories:
            image_type = "w"
        else:
            image_type = ground_truth[0]
            ground_truth = ground_truth[1:]
        image_type_entire = mapping[image_type]
        if animal in image_name:
            if condition == image_type_entire and image_name not in seen:
                human_data["image_shown"][i] = image_name.replace(".jpg", "")
                seen.append(image_name)
                break

human_data.to_csv('human_data_normal/human_data.csv')






