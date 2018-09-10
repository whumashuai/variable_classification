import data_helpers
import train
import eval

def process(input):
    data_file = data_helpers.traceHelper(input)
    print(data_file)
    train_file, eval_file = data_helpers.depart_data(data_file)
    print("train_file=", train_file)
    print("eval_file=", eval_file)
    [x_test, y, category] = data_helpers.load_data_and_labels(data_file)
    print(category)
    max_document_length = max([len(x) for x in x_test])
    if max_document_length > 2000:
        print("The length of data is too long")
    else:
        for i in range(len(category)):
            out_dir = train.train(train_file, i)
        eval.eval(eval_file, out_dir)

if __name__ == "__main__":
    # default path="F:\variable_classification\data\opencv\opencv_test_core.traces"
    data_file = 'F:\\variable_classification\\data\\bitcoin\\traces.csv'
    process(data_file)