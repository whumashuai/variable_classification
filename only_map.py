import random

in_path = r"E:\variable_classification\variable_classification\data3\no_list_template_short.txt"
out_path = r"E:\variable_classification\variable_classification\data3\no_list_template_short_map.txt"
with open(in_path, 'r') as in_file, open(out_path, 'w') as out_file:
    for line in in_file:
        sep = line[:-1].split(',')
        if sep[-1] == 'std_map':
            out_file.write(line)
        else:
            for v in sep[:-1]:
                out_file.write(v + ',')
            out_file.write('other\n')
            
