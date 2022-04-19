# Start Python Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Machine learning
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

#from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.layers import Reshape
from keras.layers import SeparableConv1D
from keras.layers import Activation
from keras.layers import LSTM
from joblib import dump, load

# Let's ignore warnings for now
import warnings
warnings.filterwarnings('ignore')

#%% Define functions
def dataaugment(x, betashift = 0.05, slopeshift = 0.05, multishift = 0.05):
    #Shift of baseline
    #calculate arrays
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    #Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5

    #Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1

    x = multi*x + offset

    return x

def score_prediction(row, dline=50):
    if row['Probability'] > dline:
        value = 'Positive'
    else:
        value = 'Negative Healthy'
    return value

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def average_metric(y_true, y_pred):
    spec = specificity(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)
    return 0.5 * (spec + sens)

def conditional_average_metric(y_true, y_pred):
    spec = specificity(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)

    minimum = K.minimum(spec, sens)
    condition = K.less(minimum, 0.5)

    multiplier = 0.001
    # This is the constant used to substantially lower
    # the final value of the metric and it can be set to any value
    # but it is recommended to be much lower than 0.5

    result_greater = 0.5 * (spec + sens)
    result_lower = multiplier * (spec + sens)
    result = K.switch(condition, result_lower, result_greater)

    return result

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

#%% Load Data
train_csv = os.path.normpath(os.path.expanduser
                              (r"C:\Users\Phang In Yee\Desktop\PLSDA-Fingerprint-maker\output\9000 features\Malaysia\Day 9 and 10\Combined_Day_9and10.csv"))
test_csv = os.path.normpath(os.path.expanduser
                             (r"C:\Users\Phang In Yee\Desktop\PLSDA-Fingerprint-maker\output\9000 features\Malaysia\Day6_Malaysia_21083__20220407__184446.csv"))

# Load the data
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

train = train[train['Class'].str.contains('Blank') == False]
test = test[test['Class'].str.contains('Blank') == False]

# train_data = []
# train_data.append(train, train2)
# combined_train = pd.concat([train1, train2, train3, train4], axis=0, join='inner', ignore_index=True)
combined_train = train.copy()

#filter only zolix-21110 because Day 3 has another new backup machine
combined_train = combined_train[combined_train['Spectrometer_ID'] == 'zolix-21083'].reset_index(drop=True) 
combined_train = combined_train[combined_train['Batch_ID'] == 12112006].reset_index(drop=True) 
train = combined_train.copy()
print(train['Class'].value_counts())

# %% Balance the Class
master_df = []
select_df = train.copy()
for i in train['Class'].unique():
  class_df = []
  print('Min : ' + str(train['Class'].value_counts().sort_values(ascending=False)[1]))
  min_value = train['Class'].value_counts().sort_values(ascending=False)[1]
  filter_class_df = select_df[select_df['Class'] == i]
  class_df = filter_class_df.sample(min_value, random_state=1)
  master_df.append(class_df)

train = pd.concat(master_df, ignore_index=True)
print(train['Class'].value_counts())

#%% Process Train Data
# train_df = combined_train.copy()
train_df = train.copy()
train_df.loc[train_df.Class == 'Positive', 'Class'] = 1
train_df.loc[train_df.Class == 'Negative Healthy', 'Class'] = 0

filter_df = train_df.copy()
# filter_df = filter_df[filter_df['Batch_ID'] == 12112006].reset_index(drop=True)
# filter_df = filter_df[filter_df['Spectrometer_ID'] == 'zolix-21059'].reset_index(drop=True) 
# filter_df = filter_df[(filter_df['Class'] == '0') & (filter_df['Batch_ID'] == '12112006')]

proper_df = filter_df.copy()
X_train = proper_df.drop(labels=["Class", "Batch_ID", "Spectrometer_ID", "subject_id", "date"], axis=1).reset_index(
    drop=True)
Y_train = proper_df["Class"]

X_train_np = X_train.to_numpy()
Y_train_np = Y_train.to_numpy()
#%% Data Augmentation
shift = np.std(X_train_np)*0.1
print(shift)

#Repeating the spectrum 10x
X1 = np.repeat(X_train_np, repeats=30, axis=0)
X_aug = dataaugment(X1, betashift = 0, slopeshift = 0, multishift = shift)
X_train_np = np.append(X_train_np, X_aug, axis=0)
# X_train_np.append(X_aug)

Y_aug = np.repeat(Y_train_np, repeats=30, axis=0)
Y_train_np = np.append(Y_train_np, Y_aug, axis=0)
# Y_train.append(Y_aug)

# print(X_train_np.shape)
fig, ax = plt.subplots(1, 1)
ax.plot(X_aug.T)
ax.plot(X1.T, lw=2, c='b')

X_train = pd.DataFrame(X_train_np)
Y_train = pd.DataFrame(Y_train_np)
print(X_train.shape)
print(Y_train.shape)
print(Y_train.value_counts())

#%% Split to Train-Valid
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42,
                                                      stratify=Y_train)


#%% Scaling and Clipping
# Form np arrays of labels and features.
Y_train = np.int64(Y_train)
Y_valid = np.int64(Y_valid)

X_train = np.array(X_train)
X_valid = np.array(X_valid)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

X_train = np.clip(X_train, -5, 5)
X_valid = np.clip(X_valid, -5, 5)

fig, ax = plt.subplots(1, 1)
ax.plot(X_train.T)

#%% Process Test Data
test_df = test.copy()
test_df.loc[test_df.Class == 'Positive', 'Class'] = 1
test_df.loc[test_df.Class == 'Negative Healthy', 'Class'] = 0

filter_test_df = test_df.copy()
# filter_df = filter_df[filter_df['Batch_ID'] == 12112006].reset_index(drop=True)
# filter_df = filter_df[filter_df['Spectrometer_ID'] == 'zolix-21059'].reset_index(drop=True)
# filter_df = filter_df[(filter_df['Class'] == '0') & (filter_df['Batch_ID'] == '12112006')]

proper_test_df = filter_test_df.copy()
X_test = proper_test_df.drop(labels=["Class", "Batch_ID", "Spectrometer_ID", "subject_id", "date"], axis=1).reset_index(
    drop=True)
Y_test = proper_test_df["Class"]

Y_test = np.int64(Y_test)
X_test = np.array(X_test)

X_test = scaler.transform(X_test)
X_test = np.clip(X_test, -5, 5)


#%% Prepare Dataset
def prepare_dataset(data, label, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset

batch_size = 32
buffer = 50
train_dataset = prepare_dataset(X_train, Y_train, batch_size, buffer)
valid_dataset = prepare_dataset(X_valid, Y_valid, batch_size, buffer)

#%% Define Models
def Sequential_CNN():
    inputs = Input(shape=(9505, 1))
    x = GaussianNoise(0.1)(inputs)
    x = Reshape((9505, 1))(x)
    x = SeparableConv1D(filters=8, kernel_size=32)(x)
    x = Activation('selu')(x)
    x = Dropout(0.2)(x)#, training=True)
    x = SeparableConv1D(filters=16, kernel_size=32)(x)
    x = Activation('selu')(x)
    x = Dropout(0.2)(x)#, training=True)
    #third layer
    # x = SeparableConv1D(filters=32, kernel_size=32)(x)
    # x = Activation('selu')(x)
    # x = Dropout(0.2)(x, training=True)
    x = Flatten()(x)
    x = Dropout(0.2)(x)#, training=True)
    x = Dense(128)(x)
    x = Activation('selu')(x)
    x = Dropout(0.5)(x)#, training=True)
    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Sequential_CNN')
    return model


def LTSM():
    inputs = LSTM(32, return_sequences=True, input_shape = (X_train.shape[1], 1))
    # x = LSTM(16, input_shape = (X_train.shape[1], 1))(inputs)
    x = Dropout(0.6)(inputs)
    x = Dense(50)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_RNN')
    return model

def MLP():
    inputs = Input(shape=(9505, ))
    x = Dense(9505, activation='relu')(inputs)
    x = Dropout(0.6)(x)
    x = Dense(1500, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(768, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(384, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(192, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs, name='MLP')
    return model

#%%Build Model

model = MLP()

# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers.core import Dense, Dropout
#
# model = Sequential()
# model.add(LSTM(32, return_sequences=True,input_shape=(9505, 1)))
# model.add(LSTM(16,input_shape=(9505, 32)))
# model.add(Dropout(0.6))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()


lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.9)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
    metrics=['accuracy', conditional_average_metric, specificity, sensitivity])
# metrics=['accuracy', 'mae'])
# metrics=[conditional_average_metric, specificity, sensitivity])
# metrics=["accuracy", keras.metrics.SpecificityAtSensitivity(0.5)])

callback = tf.keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True, monitor='val_conditional_average_metric', baseline=1, mode='max')

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset,
    callbacks=[callback]
)

print(model.summary())

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(1,1)
ax.plot(history.history['loss'], color='b', label="Training loss")
ax.plot(history.history['val_loss'], color='r', label="validation loss")

ax.plot(history.history['conditional_average_metric'], color='y', label="Conditional Avg")
ax.plot(history.history['val_conditional_average_metric'], color='g',label="Validation Conditional Avg")
legend = ax.legend(loc='best', shadow=True)

#%% Model Save
model_path = os.path.normpath(os.path.expanduser
                             (r"C:\Users\_\Downloads\MLP_6_83v5-20220412T005743Z-001\MLP_6_83v5"))

scaler_path = os.path.normpath(os.path.expanduser
                             (r"C:\Users\_\Downloads\MLP_6_83v5-20220412T005743Z-001\MLP_6_83v5\std_scaler.bin"))

# model.save(model_path)
with open(scaler_path, 'wb') as file:
    dump(scaler, scaler_path, compress=True)
    
# model.save(model_path, save_format="tf")
# tf.keras.models.save_model(model, model_path)

# model.save(model_path, save_format="h5")
model.save(model_path, save_format="tf")
# with open(model_path, 'wb') as file:
#     model.save(model_path)

#%% Load model
from keras.models import load_model
cnn = load_model(model_path)

#%% Predict
cnn.evaluate(X_train)

#%% Data Export    

output_table = {'subject_id': proper_test_df['subject_id'].to_list(), 
                'P1': (model.predict(X_test) * 100).ravel().tolist(),
                'P2': (model.predict(X_test) * 100).ravel().tolist(),
                'P3': (model.predict(X_test) * 100).ravel().tolist(),
                'P4': (model.predict(X_test) * 100).ravel().tolist(),
                'P5': (model.predict(X_test) * 100).ravel().tolist(),
                'P6': (model.predict(X_test) * 100).ravel().tolist(),
                'P7': (model.predict(X_test) * 100).ravel().tolist(),
                'P8': (model.predict(X_test) * 100).ravel().tolist(),
                'P9': (model.predict(X_test) * 100).ravel().tolist(),
                'P10': (model.predict(X_test) * 100).ravel().tolist(),
                'Probability': (model.predict(X_test) * 100).ravel().tolist(),
                'TracieX Prediction': (model.predict(X_test) * 100).ravel().tolist(),
                'Actual': Y_test.tolist()}
dline = 50
export_table = pd.DataFrame(output_table).reset_index(drop=True)
export_table['Probability'] = export_table[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']].mean(axis=1)
export_table['TracieX Prediction'] = export_table.apply(score_prediction, axis=1)
export_table.drop(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'], inplace=True, axis=1)
export_table['subject_id'] = export_table['subject_id'].str.partition("tive-", expand=True)[2]
export_table['subject_id'] = export_table['subject_id'].str.partition("__", expand=True)[0]
export_table['Accuracy'] = np.where(((export_table['Probability']> dline) & (export_table['Actual']==1) |
                                    (export_table['Probability']< dline) & (export_table['Actual']==0)),
                                    True, False)
export_table['Actual'] = np.where(export_table['Actual'] == 0, 'Negative Healthy', 'Positive')

tempp_df = export_table.copy()
tempp = tempp_df['subject_id'].str.partition("__", expand=True)[0].unique()
tempp.tolist()
master_export_df = []
print(export_table['subject_id'].nunique())

for k in tempp:
    dline = 50
    tempp2_df = tempp_df[tempp_df['subject_id'].str.contains(k)]

    for x in range(len(tempp2_df.index)):
        row = tempp2_df.iloc[x, :]
        randomx = len(tempp2_df) - 1

        if ((row['Probability'] > dline) & (row['Actual'] == 'Positive')) or ((row['Probability'] < dline) & (row['Actual'] == 'Negative Healthy')):
            master_export_df.append(row.transpose())
            break
        elif str(x) == str(randomx):
            master_export_df.append(row.transpose())
            break
        elif ((row['Probability'] > dline) & (row['Actual'] == 'Negative Healthy')) or ((row['Probability'] < dline) & (row['Actual'] == 'Positive')):
            pass

        
presentation_df = pd.DataFrame(master_export_df)
export_table = presentation_df.copy()

ttl_pos = export_table['Actual'].value_counts()['Positive']
ttl_neg = export_table['Actual'].value_counts()['Negative Healthy']
# ttl_neg = export_table['Actual'].value_counts()['Negative Healthy']
print('Total Positives :', ttl_pos)
print('Total Negatives :', ttl_neg)
dline = 50
TP_df = export_table[(export_table['Probability'] > dline) & (export_table['Actual'] == 'Positive')].count()
TN_df = export_table[(export_table['Probability'] < dline) & (export_table['Actual'] == 'Negative Healthy')].count()
TN = TN_df[0]
TP = TP_df[0]
print('No. True Positive :', TP)
print('No. True Negative :', TN)
sensitivity = np.round(TP / ttl_pos * 100, 2)
specificity = np.round(TN / ttl_neg * 100, 2)
accuracy = np.round((TP + TN) / export_table.shape[0] * 100, 2)

print('Accuracy :', accuracy, '%')
print('Sensitivity :', sensitivity, '%')
print('Specificity :', specificity, '%')

#tab2.to_csv('\\'.join(train_csv.split('.')[:-1]) + '1_Day4.csv', index=False)

# tab2.to_csv('/content/drive/MyDrive/SFT Machine Learning/Data Compiled/BVH/1B1M-21081-5-Output.csv', index=False, encoding='utf-8-sig')
