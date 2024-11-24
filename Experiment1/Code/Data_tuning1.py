from datetime import timedelta
import pandas as pd
import os
from sklearn import preprocessing
import pickle

# Define the path to the data folder
data_folder = '/Users/nickschmeiter/Downloads/NeuralNetwork/Experiment1/Data'

# Initialize an empty list to store individual dataframes
dataframes = []

# Loop through all files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate all dataframes into one
all_data_df = pd.concat(dataframes, ignore_index=True)


# Convert the 'Date' column to datetime datatype from pandas
all_data_df['Date'] = pd.to_datetime(all_data_df['Date'])
time = min(all_data_df['Date'])
end = max(all_data_df['Date'])


#convert T to timedelta object
T = timedelta(days=142)
#142 days for 100 trading days
Tindex = 100

# Adjust the path to match the correct structure
base_path = '/Users/nickschmeiter/Downloads/NeuralNetwork/Experiment1/Data/Samples'

if not os.path.exists(base_path):
    os.makedirs(base_path)  # Ensure the directory exists


while time < end - T:
    #create subset of data which is used for forcasting
    subset = all_data_df[ (all_data_df['Date'] >= time) & (all_data_df['Date'] < time + T + timedelta(days=1))]
    table = pd.pivot_table(subset, index=['Date'], columns=['Ticker'])
    table.columns = table.columns.get_level_values(1)
    table = table.dropna(axis=1)
    if table.shape[1] == 0:
        print(f"No valid data after dropna at time {time}. Skipping...")
        time = time + timedelta(days=1)
        continue
    table = table.loc[:,table.nunique()!=1]
    if table.shape[1] == 0:
        print(f"No non-constant data at time {time}. Skipping...")
        continue
    #normalize
    x = table.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    if x.shape[1] == 0:
        continue
    x_scaled = min_max_scaler.fit_transform(x)

    table = pd.DataFrame(x_scaled, columns=table.columns)
    #check for size to prevent out of bounds error
    if len(table)<=Tindex:
        Tindex = len(table)-1
    targets = table.iloc[Tindex]
    table = table[0:Tindex]
    correlations = table.corr()
    correlations = correlations.dropna(axis=1)
    correlations = correlations.loc[:, ~correlations.columns.duplicated()]
    if correlations.shape[1] == 0:
        print(f"No correlations calculated at time {time}. Skipping...")
        continue

    
    for col in correlations.columns:
        item = dict()
        item['y'] = targets[col]
        item['Y'] = table[col]
        pos_stocks = list(correlations[col].nlargest(21).index) # largest correlation is with stock itself
        pos_stocks.remove(col)
        item['X_p'] = table[pos_stocks]
        neg_stocks = list(correlations[col].nsmallest(20).index)
        item['X_n'] = table[neg_stocks]

        if len(pos_stocks) < 20 or len(neg_stocks) < 20:
            print(f"Not enough correlated stocks for {col} at time {time}. Skipping...")
            time = time + timedelta(days=1)
            continue
        # Use the base_path to construct the file path
        file_name = f"{col}_{str(time)}.pkl"
        file_path = os.path.join(base_path, file_name)
        
        with open(file_path, 'wb') as f:
            pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)
        #with open('c:/data/Samples/' + col + '_' + str(time) + '.pkl', 'wb') as f:
         #   pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)
    time = time + timedelta(days=1)