import os


max_length = 1999
tag = ',-1 -1 -1 -1 -1 -1'
fill = '-1 -1 -1 -1 -1 -1'
def write_file(data):
    print(len(data))
    print(data)

def split_length(example):
    test = []
    label = example[-1]
    example = example[:-1]
    example = ",".join(example)
    data = example.split(tag)
    if '' in data:
        data.remove('')
    for x in data:
        data1 = x.split(',')
        while '' in data1:
            data1.remove('')
        test.append(data1)
    return test,label


def process1(input,output,output1):

    files = os.listdir(input)
    for file in files:
        with open(input + '\\' + file, 'r') as f:
            for line in f:
                example = list(filter(len, map(str.strip, line.split(','))))
                if len(example)>max_length:
                    open(output1, 'a').write(",".join(example[1:]) + ',' + example[0] + '\n')
                else:
                    if len(example) < 2:
                        continue
                    open(output, 'a').write(",".join(example[1:]) + ',' + example[0] + '\n')
    return(output,output1)



def process2(output1,output2):
    examples = open(output1,'r').readlines()
    summary = 0
    data = []
    for example in examples:
        example = example.split(',')
        test,label = split_length(example)
        for x in test:
            summary = summary + len(x) + 1
            if summary < max_length:
                for a in x:
                    data.append(a)
                data.append(fill)
            else:
                open(output2, 'a').write(",".join(data) + ',' + label)
                summary = 0

                data.clear()
                for a in x:
                    data.append(a)
                data.append(fill)
    os.remove(output1)
    return output2

def process3(input,output1,output2):
    examples = open(input,'r',encoding='utf-8').readlines()
    for line in examples:
        example = list(filter(len, map(str.strip, line.split(','))))
        if len(example) > max_length:
            open(output1, 'a').write(",".join(example[1:]) + ',' + example[0] + '\n')
        else:
            if len(example) < 2:
                continue
            open(output2, 'a').write(",".join(example[1:]) + ',' + example[0] + '\n')
    output = process2(output1,output2)
    return output
if __name__ == '__main__':
    input = r"E:\variable_classification\variable_classification\path\path_bitcoin\bitcoind.no_slice.paths"
    output = r"E:\variable_classification\variable_classification\path\path_bitcoin\bitcoind.no_slice.txt"
    output1 = r"E:\variable_classification\variable_classification\path\path_bitcoin\bitcoind.no_slice1.txt"
    #output, ouput1 = process1(input,output,output1)
    #process2(output1,output)
    process3(input,output1,output)
    process2(output1,output)


