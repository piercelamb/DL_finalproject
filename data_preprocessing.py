import os
import pandas as pd

DEEPMIND_MATH_PATH = './data/mathematics_dataset-v1.0'

def get_labeled_data():
    # Possible versions ' 13 ', ' 13,', '8*13', ' 13.2008'
    # TODO update removed numbers with all possibilities
    # TODO may need a regex that matches ' 13 ', ' 13,', '8*13', ' 13.2008' but not '11113111'
    removed_numbers = [' 13 ', ' 31 ', ' 82 ', ' 99 ']
    print("Getting arthimetic data and filtering out any instances with the following numbers: "+', '.join(removed_numbers))
    interim_data = []
    count_removed = 0
    for subdir, dirs, files in os.walk(DEEPMIND_MATH_PATH):
        for file in files:
            if 'arithmetic' in file:
                full_path = os.path.join(subdir, file)
                print("Parsing: "+full_path)
                with open(full_path, 'r') as f:
                    data = f.readlines()
                    for i in range(0, len(data), 2):

                        q = data[i].replace('\n', '')
                        a = data[i+1].replace('\n', '')
                        # Test if our removed numbers appear in either the question or answer
                        if (any(num in q for num in removed_numbers)) or \
                            (any(num in a for num in removed_numbers)):
                            count_removed += 1
                        else:
                            interim_data.append([q, a])

    df = pd.DataFrame(interim_data, columns=['Question','Answer'])
    print(df)
    print("Total removed instances: "+str(count_removed))

