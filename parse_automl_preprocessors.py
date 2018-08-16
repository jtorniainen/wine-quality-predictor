import pandas as pd

if __name__ == '__main__':
    target_file = '/home/jarto/work/wine-quality-predictor/b87fe6d2-ab92-4d36-8946-111850523f12/AutoML(1):wine-quality.log'


    with open(target_file, 'r') as file_h:
        content = file_h.readlines()

    preprocessors = []
    regressors = []
    rescalers = []
    idx = 0
    for line in content:
        if 'preprocessor:__choice__' in line:
            preprocessors.append(line.split('Value: ')[1].replace("'", '').strip())
        elif 'regressor:__choice__' in line:
            regressors.append(line.split('Value: ')[1].replace("'", '').strip())
        elif 'rescaling:__choice__' in line:
            rescalers.append(line.split('Value: ')[1].replace("'", '').strip())

    # just a quick check
    df = pd.DataFrame({'preprocessor': preprocessors, 'regressor': regressors, 'rescaling': rescalers})

    for idx, row in df.iterrows():
        for idx2, row2 in df.iterrows():
            if idx == idx2:
                continue
            if row['preprocessor'] == row2['preprocessor'] and row['regressor'] == row2['regressor'] and row['rescaling'] == row2['rescaling']:
                print(row)
                print(row2)


