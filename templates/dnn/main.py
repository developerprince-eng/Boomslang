from keras.models import load_model, model_from_json
import dataset.main as dt

import os
os.getcwd()
os.listdir(os.getcwd())

def main():
    create_dataset = dt.DATASET()
    model = model_from_json('seq03_model.json')
    # model.load_weights('seq03_model.h5')
    # data = create_dataset.__read_csv__('input/Test.csv')
    # classes = model.predict(data, batch_size=32)

    # print(classes)

if __name__ == "__main__":
    main()
