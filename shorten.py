import random

in_path = r"E:\variable_classification\variable_classification\data3\no_list_template.txt"
out_path = r"E:\variable_classification\variable_classification\data3\no_list_template_short.txt"
with open(in_path, 'r') as in_file, open(out_path, 'w') as out_file:
    for line in in_file:
        sep = line[:-1].split(',')
        n = len(sep) - 1
        if n >= 50 and n <= 620 and ((sep[-1] == 'primitive' and random.randint(0,100)==0) or
                         (sep[-1] == 'std_map' and random.randint(0,1)==0) or
                         (sep[-1] == 'std_vector') or
                         (sep[-1] == 'std_list') or
                         (sep[-1] == 'std_pair') ):
            out_file.write(line)
            
