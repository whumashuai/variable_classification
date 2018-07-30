import remove_null
import data_helpers
import train
import eval
import tensorflow as tf

def process(input,output1,output2):
    data_file = remove_null.process3(input, output1, output2)
    print(data_file)
    train_file, eval_file = data_helpers.depart_data(data_file)
    print("train_file=", train_file)
    print("eval_file=", eval_file)
    [x_test, y, category] = data_helpers.load_data_and_labels(data_file)
    print(category)
    max_document_length = max([len(x) for x in x_test])
    if max_document_length > 6000:
        print("The length of data is too long")
    else:
        sum_tp, sum_fp, sum_tn, sum_fn = 0.0, 0.0, 0.0, 0.0
        for i in range(len(category)):
            tp, fp, tn, fn, out_dir = train.train(train_file, i)
            sum_tp += tp
            sum_fp += fp
            sum_tn += tn
            sum_fn += fn
        precision = tf.div(tp, tf.add(tp, fp))
        recall = tf.div(tp, tf.add(tp, fn))
        # F1-score
        product = tf.multiply(tf.cast(tf.constant(2.0), tf.float64), tf.multiply(precision, recall))
        f1_score = tf.div(product, tf.add(precision, recall))
        with open('./result.txt', 'w') as f:
            f.write("recall {:g}, f1_score {:g}, precision {:g}".format(recall, f1_score, precision))

        accuracy = eval.eval(eval_file, out_dir)
        with open('./result.txt', 'a') as f:
            f.write("accuracy {:g}".format(accuracy))
if __name__ == "__main__":
    input = './path/path_bitcoin/bitcoind.no_slice.paths'
    output1 = './path/path_bitcoin/bitcoind.no_slice1.txt'
    output2 = './path/path_bitcoin/bitcoind.no_slice.txt'
    process (input, output1, output2)