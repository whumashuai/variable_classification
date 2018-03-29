import random

in_path = r"C:\Users\wust_\OneDrive\Documents\Tencent Files\1459686989\FileRecv\lli_paths.txt"
out_path = r"E:\variable_classification\variable_classification\llvm_as_path\llvm-as_paths_san.txt"

labelset = set()
with open(in_path, 'r') as in_file, open(out_path, 'w') as out_file:
    for line in in_file:
        sep = line[:-1].split(',')
        labelset.add(sep[-1])
        #if sep[-1] == 'other_template':
        #    continue
        #out_file.write(line)


print(labelset)
