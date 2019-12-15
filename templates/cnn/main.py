from keras.models import load_model, model_from_json

import os
os.getcwd()
os.listdir(os.getcwd())

def main():
    model = model_from_json('CNN01x_model.json')
    # model.load_weights('seq03_model.h5')
    # data = create_dataset.__read_csv__('input/Test.csv')
    # classes = model.predict(data, batch_size=32)

    # print(classes)

if __name__ == "__main__":
    main()
