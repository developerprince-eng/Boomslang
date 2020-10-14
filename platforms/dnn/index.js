/*
PROJECT TITLE           : BoomSlang-AI-CLI
PROJECT ALIAS           : fluid-serpent
PROJECT OWNER           : DeveloperPrince
PROJECT LEAD            : DeveloperPrince (Prince Kudzai Maposa)
AFFILIATION             : Semina
PROJECT BRIEF           : This is command line interface which will enable developers to scaffold 
                          project of any nature which are ai based, currently focusing 
                          python.

Happy Coding 
*/
/*--------------Modules-------------------*/
const cp = require(`child_process`)
const fs = require(`fs`)
const isWin = process.platform === "win32"
const internetAvailable = require(`internet-available`)
const { stdout } = require("process")

let dnn = (name) => {

    let winEnv = () => {
        cp.execSync(`mkdir ${name} `, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`CREATED FOLDER:  ${name} & Switched to Root Project Folder`)
            console.log(stderr)
        })
       
        cp.execSync(`cd ${name} & type nul > run.py & mkdir dataset & mkdir model & cd model & type nul > main.py & cd ..`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Created Core Files to  ${name} Folder`)
            console.log(stderr)
        })

        writeFiles(name)
        cp.exec(`cd ${name} & git init && git add . & git commit -m "Initial Commit, Using BoomSlangCLI" `, (err, stdout, stderr)=>{
            if(err) throw err
            if(stdout) {
                webScaf(stdout)
                websuccess(stdout)
                dpLogo(stdout)
            }
            if(stderr) console.log(stderr)
        })
    }

    let dpLogo = (stdout) => {
        console.log(`   &&&&     &&&&&&&&&&&&&&&&&&&&&&&     &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&        \n   &&&&     &&&&&&&&&&&&&&&&&&&&&&&     &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&        \n   &&&&                       #&&&&     &&&&                       &&&&&        \n   &&&&                       #&&&&     &&&&                       &&&&&        \n   &&&&&&&&&&&&&&&&&&&&&&&    #&&&&     &&&&     &&&&&&&&&&&&&&&&&&&&&&&        \n   &&&&&&&&&&&&&&&&&&&&&&&    #&&&&     &&&&     &&&&&&&&&&&&&&&&&&&&&&&        \n                     &&&&&    #&&&&     &&&&     &&&&%                          \n                     &&&&&    #&&&&     &&&&     &&&&%                          \n   &&&&&&&&&&&&&&&&&&&&&&&    #&&&&     &&&&     &&&&%    &&&&&&&&&&&&&&        \n   &&&&&&&&&&&&&&&&&&&&&&&    #&&&&     &&&&     &&&&%    &&&&&&&&&&&&&&        \n   &&&&                       #&&&&     &&&&     &&&&%    &&&&&    &&&&&        \n   &&&&     \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/%&&&&     &&&&     &&&&%    &&&&&    &&&&&        \n   &&&&     &&&&&&&&&&&&&&&&&&&&&&&     &&&&     &&&&%    &&&&&    &&&&&    &&&`)
    }

    //Success Console Message for Web Scaffolding
    let webScaf = (stdout) =>{
        console.log(
            `****************************************\n`,
            `#####Lets Switch up things a little#####\n`,
            `****************************************\n`,
            `Now Lets Do Magic using Hub to Access Github\n`,
            `****************************************\n`,
            `Scaffolding: \n\x1b[33m`, stdout,
            `****************************************\n`,
            `WE Have Successfully Pushed The Project \n`,
            `______________^(* _ *)^________________\n`,
            `************* ENJOY!!! *****************\n`,
            `************* HAPPY CODING *************`)
    }

    let webSuccess = (name)=> {
        console.log(`\x1b[36m%s\x1b[0m`, name, `Project Scaffolded Successfully...\n`,`\x1b[33m`,`Enjoy!!!`,`\x1b[0m`);
    } 
    let unixEnv = () =>{
        cp.execSync(`mkdir ${name} `, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`CREATED FOLDER:  ${name} `)
            console.log(stderr)
        })

        cp.execSync(`cd ${name}`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Switched to Root Project Folder`)
            console.log(stderr)
        })
       
        cp.execSync(`cp -r /usr/lib/node_modules/boomslang-cli/templates/dnn/ ${name}`, (err, stdout, stderr) => {
            if(err) throw err
            if(stdout) console.log(`Copied Core Files to  ${name} Folder`)
            console.log(stderr)
        })

        cp.exec(`cd ${name} & git init && git add . & git commit -m "Initial Commit, Using BoomSlangCLI" `, (err, stdout, stderr)=>{
            if(err) throw err
            if(stdout) {
                webscaf(stdout)
                websuccess(stdout)
            }
            if(stderr) console.log(stderr)
        })
    }

    let writeFiles = (name) => {
        const modelContent = 
`
from __future__ import absolute_import, division, print_function\n
from bokeh.plotting import figure, output_file, show\n
from keras.models import Sequential\n
from keras.layers import Dense\n
from keras import regularizers\n
from keras.callbacks import ModelCheckpoint\n
import matplotlib.pyplot as plt\n
import matplotlib.pyplot as plt2\n
# import tensorflow as tf currently no support for tensorflow manually install(make sure pc  supports tensorflow)\n
import pandas as pd\n
import os\n
from os.path import realpath, abspath\n
import numpy as np \n\n
os.getcwd()\n
os.listdir(os.getcwd())\n
class MODEL():\n
\tdef __init__(self):\n
\t\tself.is_model = True\n\n
\t#Train RNN Classifier\n
\tdef kr_train_RNN_Model(self):\n
\t\tmetrics = []\n
\t\treturn metrics\n
\tdef kr_train_DNN_Seq_01(self,x_dim ,features_train ,features_test, labels_train , labels_test, batch_size):\n
\t\t# create model\n
\t\tmodel = Sequential()\n
\t\tmodel.add(Dense(10, input_dim=x_dim, init='uniform', activation='relu'))\n
\t\tmodel.add(Dense(60, init='uniform', activation='relu'))\n
\t\tmodel.add(Dense(30, init='uniform', activation='sigmoid'))\n
\t\tmodel.add(Dense(1, init='uniform', activation='sigmoid'))\n
\t\t# Compile model\n
\t\tmodel.compile(loss='mean_squared_logarithmic_error', optimizer='adam',\n
\t\tmetrics=['accuracy'])\n\n
\t\t# checkpoint\n
\t\tfilepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"\n
\t\tcheckpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')\n
\t\tcallbacks_list = [checkpoint]\n
\t\t# Model Summary\n
\t\tmodel.summary()\n
\t\t# Fit the model\n
\t\tmodel.fit(features_train, labels_train, epochs=100, batch_size=batch_size, verbose=2)\n
\t\tmodel_yaml = model.to_yaml()\n
\t\twith open("seq01_model.yaml", "w") as yaml_file:\n
\t\t\tyaml_file.write(model_yaml)\n
\t\t# serialize weights to HDF5\n
\t\tmodel_json = model.to_json()\n
\t\twith open("seq01_model.json", "w") as json_file:\n
\t\t\tjson_file.write(model_json)\n
\t\tmodel.save_weights("seq01_model.h5")\n
\t\tscore = model.evaluate(features_test, labels_test, verbose=1)\n
\t\t# round predictions\n
\t\taccuracy = score[1]\n
\t\taccuracy = accuracy * 100\n\n
\t\t#Plot training & validation accuracy values\n
\t\tplt.plot(history.history['accuracy'])\n
\t\tplt.plot(history.history['val_accuracy'])\n
\t\tplt.title('Model accuracy')\n
\t\tplt.ylabel('Accuracy')\n
\t\tplt.xlabel('Epoch')\n
\t\tplt.legend(['Train', 'Test'], loc='upper left')\n
\t\tplt.show()\n\n
\t\t#Plot training & validation loss values\n
\t\tplt2.plot(history.history['loss'])\n
\t\tplt2.plot(history.history['val_loss'])\n
\t\tplt2.title('Model loss')\n
\t\tplt2.ylabel('Loss')\n
\t\tplt2.xlabel('Epoch')\n
\t\tplt2.legend(['Train', 'Test'], loc='upper left')\n
\t\tplt2.show()\n\n
\t\treturn accuracy\n\n\n
\tdef kr_train_DNN_Seq_02(self,x_dim ,features_train ,features_test, labels_train , labels_test, batch_size):\n
\t\t# create model\n
\t\tmodel = Sequential()\n
\t\tmodel.add(Dense(9, input_dim=x_dim, init='uniform', activation='relu'))\n
\t\tmodel.add(Dense(36, init='uniform', activation='relu'))\n
\t\tmodel.add(Dense(36, init='uniform', activation='sigmoid'))\n
\t\tmodel.add(Dense(1, init='uniform', activation='sigmoid'))\n
\t\t# Compile model\n
\t\tmodel.compile(loss='mean_squared_error', optimizer='adam',\n
\t\tmetrics=['accuracy'])\n\n
\t\t# checkpoint\n
\t\tfilepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"\n
\t\tcheckpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')\n
\t\tcallbacks_list = [checkpoint]\n
\t\t# Model Summary\n
\t\tmodel.summary()\n
\t\t# Fit the model\n
\t\tmodel.fit(features_train, labels_train, epochs=100, batch_size=batch_size, verbose=2)\n
\t\tmodel_yaml = model.to_yaml()\n
\t\twith open("seq02_model.yaml", "w") as yaml_file:\n
\t\t\tyaml_file.write(model_yaml)\n
\t\t# serialize weights to HDF5\n
\t\tmodel_json = model.to_json()\n
\t\twith open("seq02_model.json", "w") as json_file:\n
\t\t\tjson_file.write(model_json)\n
\t\tmodel.save_weights("seq02_model.h5")\n
\t\tscore = model.evaluate(features_test, labels_test, verbose=1)\n
\t\t# round predictions\n
\t\taccuracy = score[1]\n
\t\taccuracy = accuracy * 100\n\n
\t\t#Plot training & validation accuracy values\n
\t\tplt.plot(history.history['accuracy'])\n
\t\tplt.plot(history.history['val_accuracy'])\n
\t\tplt.title('Model accuracy')\n
\t\tplt.ylabel('Accuracy')\n
\t\tplt.xlabel('Epoch')\n
\t\tplt.legend(['Train', 'Test'], loc='upper left')\n
\t\tplt.show()\n\n
\t\t#Plot training & validation loss values\n
\t\tplt2.plot(history.history['loss'])\n
\t\tplt2.plot(history.history['val_loss'])\n
\t\tplt2.title('Model loss')\n
\t\tplt2.ylabel('Loss')\n
\t\tplt2.xlabel('Epoch')\n
\t\tplt2.legend(['Train', 'Test'], loc='upper left')\n
\t\tplt2.show()\n\n
\t\treturn accuracy\n\n\n
\tdef kr_train_DNN_Seq_03(self,x_dim ,features_train ,features_test, labels_train , labels_test, batch_size):\n
\t\t# create model\n
\t\tmodel = Sequential()\n
\t\tmodel.add(Dense(28, input_dim=x_dim, init='uniform',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.005), activation='relu'))\n
\t\tmodel.add(Dense(14, init='uniform',kernel_regularizer=regularizers.l2(0.015),activity_regularizer=regularizers.l1(0.02), activation='relu'))\n
\t\tmodel.add(Dense(8, init='uniform', activation='relu'))\n
\t\tmodel.add(Dense(10, init='uniform', activation='relu'))\n
\t\tmodel.add(Dense(10, init='uniform', activation='relu'))\n
\t\tmodel.add(Dense(1, init='uniform', activation='relu'))\n
\t\t# Compile model\n
\t\tmodel.compile(loss='mean_squared_logarithmic_error', optimizer='adam',\n
\t\tmetrics=['accuracy'])\n\n
\t\t# checkpoint\n
\t\tfilepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"\n
\t\tcheckpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')\n
\t\tcallbacks_list = [checkpoint]\n
\t\t# Model Summary\n
\t\tmodel.summary()\n
\t\t# Fit the model\n
\t\tmodel.fit(features_train, labels_train, epochs=100, batch_size=batch_size, verbose=2)\n
\t\tmodel_yaml = model.to_yaml()\n
\t\twith open("seq03_model.yaml", "w") as yaml_file:\n
\t\t\tyaml_file.write(model_yaml)\n
\t\t# serialize weights to HDF5\n
\t\tmodel_json = model.to_json()\n
\t\twith open("seq03_model.json", "w") as json_file:\n
\t\t\tjson_file.write(model_json)\n
\t\tmodel.save_weights("seq02_model.h5")\n
\t\tscore = model.evaluate(features_test, labels_test, verbose=1)\n
\t\t# round predictions\n
\t\taccuracy = score[1]\n
\t\taccuracy = accuracy * 100\n\n
\t\t#Plot training & validation accuracy values\n
\t\tplt.plot(history.history['accuracy'])\n
\t\tplt.plot(history.history['val_accuracy'])\n
\t\tplt.title('Model accuracy')\n
\t\tplt.ylabel('Accuracy')\n
\t\tplt.xlabel('Epoch')\n
\t\tplt.legend(['Train', 'Test'], loc='upper left')\n
\t\tplt.show()\n\n
\t\t#Plot training & validation loss values\n
\t\tplt2.plot(history.history['loss'])\n
\t\tplt2.plot(history.history['val_loss'])\n
\t\tplt2.title('Model loss')\n
\t\tplt2.ylabel('Loss')\n
\t\tplt2.xlabel('Epoch')\n
\t\tplt2.legend(['Train', 'Test'], loc='upper left')\n
\t\tplt2.show()\n\n
\t\treturn accuracy\n\n\n
`
        const runContent = 
`
from keras.models import load_model, model_from_json\n
import dataset.main as dt\n\n
import os\n\n
os.getcwd()\n
os.listdir(os.getcwd())\n\n
def main():\n
\tcreate_dataset = dt.DATASET()\n
\tmodel = model_from_json('seq03_model.json')\n
\t# model.load_weights('seq03_model.h5')\n
\t# data = create_dataset.__read_csv__('input/Test.csv')\n
\t# classes = model.predict(data, batch_size=32)\n\n
\t# print(classes)\n\n
if __name__ == "__main__":\n
\tmain()\n
`

        const dataContent = 
`
from __future__ import absolute_import, division, print_function\n\n
import pandas as pd \n
import os\n
from os.path import realpath, abspath\n
import numpy as np \n
from sklearn.preprocessing import LabelEncoder, OneHotEncoder\n
from sklearn.model_selection import train_test_split\n\n
os.getcwd()\n
os.listdir(os.getcwd())\n\n
class DATASET():\n
\tdef __init__(self):\n
\t\tself.is_data = True\n\n
\tdef __read_csv__(self, path):\n
\t\tdata_set = pd.read_csv(path, low_memory=False)\n
\t\treturn data_set\n\n
\tdef __obtain_data__csv__el__(self, path, number_features, number_labels, test_size=None, random_state=None):\n
\t\tif (test_size is not None and random_state is not None):\n
\t\t\tdata_set = pd.read_csv(path, low_memory=False)\n
\t\t\tinput_x = data_set.iloc[ : , 0:(number_features)]\n
\t\t\tinput_y = data_set.iloc[ : , number_features:(number_features + number_labels)]\n\n
\t\t\tx_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = test_size, random_state = random_state)\n
\t\t\treturn input_x, x_train, x_test, y_train, y_test\n\n
\t\telif (random_state is not None):\n
\t\t\tdata_set = pd.read_csv(path, low_memory=False)\n
\t\t\tinput_x = data_set.iloc[ : , 0:(number_features)]\n
\t\t\tinput_y = data_set.iloc[ : , number_features:(number_features + number_labels)]\n\n
\t\t\tx_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = 0.2, random_state = random_state)\n\n
\t\t\treturn input_x, x_train, x_test, y_train, y_test\n\n
\t\telse:\n
\t\t\tdata_set = pd.read_csv(path, low_memory=False)\n
\t\t\tinput_x = data_set.iloc[ : , 0:(number_features)]\n
\t\t\tinput_y = data_set.iloc[ : , number_features:(number_features + number_labels)]\n\n
\t\t\tx_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = 0.2, random_state = 0)\n\n
\t\t\treturn input_x, x_train, x_test, y_train, y_test\n\n
\tdef __obtain_data__csv__fl__(self, path, number_features, number_labels, test_size=None, random_state=None):\n
\t\tif (test_size is not None and random_state is not None):\n
\t\t\tdata_set = pd.read_csv(path, low_memory=False)\n
\t\t\tinput_x = data_set.iloc[ : , (number_labels+1):(number_features+number_labels+1)]\n
\t\t\tinput_y = data_set.iloc[ : , 1:(number_labels+1)]\n\n
\t\t\tx_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = test_size, random_state = random_state)\n
\t\t\treturn input_x, x_train, x_test, y_train, y_test\n\n
\t\telif (random_state is not None):\n
\t\t\tdata_set = pd.read_csv(path, low_memory=False)\n
\t\t\tinput_x = data_set.iloc[ : , (number_labels+1):(number_features+number_labels+1)]\n
\t\t\tinput_y = data_set.iloc[ : , 1:(number_labels+1)]\n
\t\t\tx_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = 0.2, random_state = random_state)\n\n
\t\t\treturn input_x, x_train, x_test, y_train, y_test\n\n
\t\telse:\n
\t\t\tdata_set = pd.read_csv(path, low_memory=False)\n
\t\t\tinput_x = data_set.iloc[ : , (number_labels+1):(number_features+number_labels+1)]\n
\t\t\tinput_y = data_set.iloc[ : , 1:(number_labels+1)]\n\n
\t\t\tx_train, x_test, y_train, y_test = train_test_split(input_x, input_y ,test_size = 0.2, random_state = 0)\n\n
\t\t\treturn input_x, x_train, x_test, y_train, y_test\n\n
`

        const modelFileStream = fs.createWriteStream(`${name}/model/main.py`)
        modelFileStream.once(`open`, function(fd){
            modelFileStream.write(modelContent)
            modelFileStream.end()
        })

        const dataFileStream = fs.createWriteStream(`${name}/dataset/main.py`)
        dataFileStream.once(`open`, function(fd){
            dataFileStream.write(dataContent)
            dataFileStream.end()
        })
        
        const runFileStream = fs.createWriteStream(`${name}/run.py`)
        runFileStream.once(`open`, function(fd){
            runFileStream.write(runContent)
            runFileStream.end()
        })
    }
    //CHECK FOR SYSTEM ENVIRONMENT
    if(isWin){
        winEnv()
    }
    else{
        unixEnv()
    }
}

exports.nueralnet = {dnn}