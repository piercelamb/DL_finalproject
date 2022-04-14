import os
import pandas as pd
import re

DEEPMIND_MATH_PATH = './data/mathematics_dataset-v1.0'


def get_labeled_data():
    # Possible versions ' 13 ', ' 13,', '8*13', ' 13.2008'
    # TODO update removed numbers with all possibilities
    # TODO may need a regex that matches ' 13 ', ' 13,', '8*13', ' 13.2008' but not '11113111'
    removed_numbers_ints = [' 13 ', ' 31 ', ' 82 ', ' 99 ']
    removed_numbers = ["\D1[3]{1}\D", "\D3[1]{1}\D", "\D8[2]{1}\D", "\D9[9]{1}\D"]
    print("Getting arthimetic data and filtering out any instances with the following numbers: " + ', '.join(
        removed_numbers_ints))
    interim_data = []
    eliminated_data = []
    count_removed = 0
    for subdir, dirs, files in os.walk(DEEPMIND_MATH_PATH):
        for file in files:
            if 'arithmetic' in file:
                full_path = os.path.join(subdir, file)
                print("Parsing: " + full_path)
                with open(full_path, 'r') as f:
                    data = f.readlines()
                    for i in range(0, len(data), 2):

                        q = data[i].replace('\n', '')
                        a = data[i + 1].replace('\n', '')
                        # Test if our removed numbers appear in either the question or answer
                        x = False
                        # Loop trough all of the regular expressions, check both the question and answer
                        for number in removed_numbers:
                            q1 = re.findall(number, q)
                            a1 = re.findall(number, a)
                            if a1 or q1: # if regex found a match in the question or answer break and change x to True
                                x = True
                                break
                        if x is True:
                            count_removed += 1
                            eliminated_data.append([q, a]) # save the eliminated data
                        else:
                            interim_data.append([q, a]) # Save the "training" data

    df = pd.DataFrame(interim_data, columns=['Question', 'Answer'])
    pdf = pd.DataFrame(eliminated_data, columns=['Question', 'Answer'])
    print(df)
    print(pdf)
    print("Total removed instances: " + str(count_removed))
