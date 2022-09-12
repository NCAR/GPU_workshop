from pathlib import Path

import numpy as np, pandas as pd, pprint
import matplotlib.pyplot as plt

#%load_ext nb_black                # Nice for iPython, not available for Colab.
#%matplotlib inline                # For iPython

import matplotlib.pyplot as plt
# Matplotlib Configuration
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}
plt.rc('font', **font)

# From our time range table in the "Data Notes" section:
period_ranges = {
    'train_a':[pd.Timestamp('1998/2/16 00:00:00'), pd.Timestamp('2001/5/31  23:59:00')], 
    'train_b':[pd.Timestamp('2013/6/1  00:00:00'), pd.Timestamp('2019/5/31  23:59:00')],
    'train_c':[pd.Timestamp('2004/5/1  00:00:00'), pd.Timestamp('2010/12/31 23:59:00')],
    'test_a' :[pd.Timestamp('2001/6/1  00:00:00'), pd.Timestamp('2004/4/30  23:59:00')],
    'test_b' :[pd.Timestamp('2011/1/1  00:00:00'), pd.Timestamp('2013/5/31  23:59:00')],
    'test_c' :[pd.Timestamp('2019/6/1  00:00:00'), pd.Timestamp('2020/10/31 23:59:00')]}

# Import as Pandas DataFrames
DATA_PATH = Path("data/public/")

print('Reading in the Dst output data...')
dst = pd.read_csv(DATA_PATH / "dst_labels.csv")
dst.timedelta = pd.to_timedelta(dst.timedelta)
dst.set_index(["period", "timedelta"], inplace=True)

print('Reading in the Sunspot input data...')
sunspots = pd.read_csv(DATA_PATH / "sunspots.csv")
sunspots.timedelta = pd.to_timedelta(sunspots.timedelta)
sunspots.set_index(["period", "timedelta"], inplace=True)

print('Reading in the Solarwind input data...')
solar_wind = pd.read_csv(DATA_PATH / "solar_wind.csv")
solar_wind.timedelta = pd.to_timedelta(solar_wind.timedelta)
solar_wind.set_index(["period", "timedelta"], inplace=True)

print('Reading in the Satellite position input data...')
satellite_positions = pd.read_csv(DATA_PATH / "satellite_positions.csv")
satellite_positions.timedelta = pd.to_timedelta(satellite_positions.timedelta)
satellite_positions.set_index(["period", "timedelta"], inplace=True)

# Set a specified seed to ensure reproducibility
from numpy.random import seed
from tensorflow.random import set_seed

seed(2020)
set_seed(2021)

# subset of solar wind features to use for modeling
SOLAR_WIND_FEATURES = [
    "bt",
    "temperature",
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "speed",
    "density",
]

# The model will be built on feature statistics, mean and standard deviation
XCOLS = (
    [col + "_mean" for col in SOLAR_WIND_FEATURES]
    + [col + "_std" for col in SOLAR_WIND_FEATURES]
    + ["smoothed_ssn"]
)

from sklearn.preprocessing import StandardScaler

def impute_features(feature_df):
    """Imputes data using the following methods:
    - `smoothed_ssn`: forward fill
    - `solar_wind`: interpolation
    """
    # forward fill sunspot data for the rest of the month
    feature_df.smoothed_ssn = feature_df.smoothed_ssn.fillna(method="ffill")
    # interpolate between missing solar wind values
    feature_df = feature_df.interpolate()
    return feature_df


def aggregate_hourly(feature_df, aggs=["mean", "std"]):
    """Aggregates features to the floor of each hour using mean and standard deviation.
    e.g. All values from "11:00:00" to "11:59:00" will be aggregated to "11:00:00".
    """
    # group by the floor of each hour use timedelta index
    agged = feature_df.groupby(
        ["period", feature_df.index.get_level_values(1).floor("H")]
    ).agg(aggs)
    # flatten hierachical column index
    agged.columns = ["_".join(x) for x in agged.columns]
    return agged


def preprocess_features(solar_wind, sunspots, scaler=None, subset=None):
    """
    Preprocessing steps:
        - Subset the data
        - Aggregate hourly
        - Join solar wind and sunspot data
        - Scale using standard scaler
        - Impute missing values
    """
    # select features we want to use
    if subset:
        solar_wind = solar_wind[subset]

    # aggregate solar wind data and join with sunspots
    hourly_features = aggregate_hourly(solar_wind).join(sunspots)

    # subtract mean and divide by standard deviation
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(hourly_features)

    normalized = pd.DataFrame(
        scaler.transform(hourly_features),
        index=hourly_features.index,
        columns=hourly_features.columns,
    )

    # impute missing values
    imputed = impute_features(normalized)

    # we want to return the scaler object as well to use later during prediction
    return imputed, scaler

features, scaler = preprocess_features(solar_wind, sunspots, subset=SOLAR_WIND_FEATURES)

YCOLS = ["t0", "t1"]


def process_labels(dst):
    y = dst.copy()
    y["t0"] = y.groupby("period").dst.shift( 0)
    y["t1"] = y.groupby("period").dst.shift(-1)
    return y[YCOLS]


labels = process_labels(dst)

data = labels.join(features)

def get_train_test_val(data, test_per_period, val_per_period):
    """Splits data across periods into train, test, and validation"""
    # assign the last `test_per_period` rows from each period to test
    test = data.groupby("period").tail(test_per_period)
    interim = data[~data.index.isin(test.index)]
    # assign the last `val_per_period` from the remaining rows to validation
    val = interim.groupby("period").tail(val_per_period)
    # the remaining rows are assigned to train
    train = interim[~interim.index.isin(val.index)]
    return train, test, val


train, test, val = get_train_test_val(data, test_per_period=6_000, val_per_period=3_000)

ind = [0, 1, 2]
names = ["train_a", "train_b", "train_c"]
width = 0.75
train_cnts = [len(df) for _, df in train.groupby("period")]
val_cnts = [len(df) for _, df in val.groupby("period")]
test_cnts = [len(df) for _, df in test.groupby("period")]

create_new_model = True     # Set True to skip the next section and create a new model.
                            # Set False to load a pre-trained model.

import tensorflow.keras as keras

import glob
# List existing LSTM models:
dir_list = glob.glob('trained_models_lstm/model_lstm_*/')
print('Here is a list of pre-trained models:\n')
for i in range(len(dir_list)):
    print('    %d: %s' % (i, dir_list[i]))

if not create_new_model:
    dir_model = dir_list[int(input('Enter number of pre-trained model: '))]
    
    import json
    import pickle
    
    # Load in serialized model, config, and scaler
    print('Loading pre-trained model from: %s' % dir_model)
    model = keras.models.load_model(dir_model)
    model.summary()

    # Load Scaler
    with open(dir_model+"/scaler.pck", "rb") as f:
        scaler = pickle.load(f)
    print('Scaler:')
    pprint.pprint(scaler)

    # Load History
    with open(dir_model+"/history.pck", "rb") as f:
        history = pickle.load(f)
    print('History:')
    pprint.pprint(history)

    # Load Configuration
    with open(dir_model+"/config.json", "r") as f:
        data_config = json.load(f)
    print('Configuration:')
    pprint.pprint(data_config)

# If we're Defining and Training a New Model
if create_new_model:
    from keras.layers import Dense, LSTM
    from keras.models import Sequential

    # Define our model
    # TODO: Adjust batch size to inspect impact on model performance
    data_config = {
        "timesteps": 32,
        "batch_size": 32,
    }
    print('data_config = ')
    pprint.pprint(data_config)

    # Hyper Parameter Tuning
    #
    # Going Big (takes hours):
    #      model_config = {"n_epochs": 30, "n_neurons": 2048, "dropout": 0.4, "stateful": False}
    #
    # Original from MagNet blogpost benchmark, takes about 1.5 hours:
    #      model_config = {"n_epochs": 20, "n_neurons": 512, "dropout": 0.4, "stateful": False}
    #
    # Takes 10-15 minutes (moderate performance): 
    #      model_config = {"n_epochs": 8, "n_neurons": 64, "dropout": 0.4, "stateful": False}
    #
    # Takes 20 seconds (anticipate bad performance):
    # TODO: Set 1 epoch to minimize length of profile runs. Try different network sizes, ie n_neurons, for additional problem sizes to profile
    model_config = {"n_epochs": 4, "n_neurons": 16, "dropout": 0.4, "stateful": False}

    model = Sequential()
    model.add(
        LSTM(
            model_config["n_neurons"],
            # usually set to (`batch_size`, `sequence_length`, `n_features`)
            # setting the batch size to None allows for variable length batches
            batch_input_shape=(None, data_config["timesteps"], len(XCOLS)),
            stateful=model_config["stateful"],
            dropout=model_config["dropout"],
        )
    )
    model.add(Dense(len(YCOLS)))
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        run_eagerly=None,     # set to True for debugging (very slow), None or False
    )

    model.summary()
    
import tensorflow.keras as keras
from keras import preprocessing

def timeseries_dataset_from_df(df, batch_size):
    dataset = None
    timesteps = data_config["timesteps"]

    # iterate through periods
    for _, period_df in df.groupby("period"):
        # realign features and labels so that first sequence of 32 is aligned with the 33rd target
        inputs = period_df[XCOLS][:-timesteps]
        outputs = period_df[YCOLS][timesteps:]

        period_ds = keras.preprocessing.timeseries_dataset_from_array(
            inputs,
            outputs,
            timesteps,
            batch_size=batch_size,
        )

        if dataset is None:
            dataset = period_ds
        else:
            dataset = dataset.concatenate(period_ds)
        
    return dataset

train_ds = timeseries_dataset_from_df(train, data_config["batch_size"])
val_ds   = timeseries_dataset_from_df(val,   data_config["batch_size"])
test_ds  = timeseries_dataset_from_df(test,  data_config["batch_size"])

print(f"Number of training batches: {len(train_ds)}")
print(f"Number of validation batches: {len(val_ds)}")
print(f"Number of test batches: {len(test_ds)}")

# TODO: Add callback function for TensorBoard profiler GUI

### Run training for new models ###
# TODO: Specify callbacks in `model.fit()`
if create_new_model:
    history_keras = model.fit(
        train_ds,
        batch_size=data_config["batch_size"],
        epochs=model_config["n_epochs"],
        verbose=True,
        shuffle=False,
        validation_data=val_ds
    )
    history = history_keras.history     # Convert Keras 'history' callback reference to a simple dictionary.
                                        # This will make saving and loading as a pickle easier, i.e. while
                                        # the TAI4ES Jupyter "GPU" environment did not have trouble saving/pickling 
                                        # as a weak reference to the keras object, the "CPU" environment gave
                                        # error "TypeError: cannot pickle 'weakref' object".
                    
for name, values in history.items():
    plt.plot(values, 's-', label=name)
plt.legend(fontsize=14)
plt.show()

rmse = model.evaluate(test_ds)**0.5
print(f"Test RMSE: {rmse:.2f} nano-Tesla")