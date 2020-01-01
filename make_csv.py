import os
import pandas as pd
from utils import return_type, load_typ_yml


train_file = sorted(os.listdir('data/train'))
test_file = sorted(os.listdir('data/test'))
type_dic = load_typ_yml()

def make_blank_csv():
    train, test = pd.DataFrame(), pd.DataFrame()
    train['id'], test['id'] = train_file, test_file
    for c in type_dic.keys():
        train[c] = 0
        test[c] = 0
    return train, test


def make_csv(df):
    for i, img in enumerate(df['id']):
        typ = return_type(img, type_dic)
        if len(typ)==1:
            df.iloc[i, typ[0]+1] = 1
        else:
            df.iloc[i, typ[0]+1] = 1
            df.iloc[i, typ[1]+1] = 1
    return df


def main():
    train, test = make_blank_csv()
    train = make_csv(train)
    test = make_csv(test)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)


if __name__ == "__main__":
    main()