from lib.Model import Model
import numpy as np
from tqdm import tqdm
import pandas


def prepare_data(df):
    x = []
    y = []

    for d in tqdm(df.values):
        # age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
        age = 0
        if d[0] <= 48:
            age = 1
        elif 48 < d[0] <= 57:
            age = 2
        sex = d[1]
        cp = d[2]
        trestbps = 0
        if d[3] <= 125:
            trestbps = 1
        elif 125 < d[3] <= 147:
            trestbps = 2
        chol = 0
        if d[4] <= 213:
            chol = 1
        elif 213 < d[4] <= 301:
            chol = 2
        fbs = d[5]
        restecg = d[6]
        thalach = 0
        if d[7] <= 123:
            thalach = 1
        elif 123 < d[7] <= 162:
            thalach = 2
        exang = d[8]
        oldpeak = d[9]
        slope = d[10]
        ca = d[11]
        thal = d[12]
        label = d[13]
        target_array = np.array([1, 0]) if label == 1 else np.array([0, 1])
        input_array = np.array([age, sex, cp, fbs, trestbps, chol, restecg, exang, oldpeak,
                                slope, ca, thal, thalach])
        x.append(input_array)
        y.append(target_array)

    return x, y


if __name__ == '__main__':
    df_read = pandas.read_csv("./heart_disease.csv")
    x_train, y_train = prepare_data(df_read)
    total = len(x_train)
    test_num = total / 5
    train_num = total - test_num

    x_test = x_train
    y_test = y_train

    model = Model([[13], [6, "relu"], [2, "softmax"]], loss_function="cross_entropy", learning_rate=0.001)

    model.train("gradient_descent", np.array(x_train), np.array(y_train), epochs=50, shuffle_data=True, batch_size=1)

    print(model.calculate_accuracy(inputs=x_test, labels=y_test))

    # model.save("../../saved/heart_disease/", "saved.pck")
