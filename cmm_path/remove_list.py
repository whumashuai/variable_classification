import random

in_path = r"E:\variable_classification\variable_classification\cmm_path\cmm_path.txt"
out_path = r"E:\variable_classification\variable_classification\cmm_path\cmm_paths.txt"
with open(in_path, 'r') as in_file, open(out_path, 'w') as out_file:
    for line in in_file:
        sep = line[:-1].split(',')
        if sep[-1] == 'other_template':
            continue
        out_file.write(line)
