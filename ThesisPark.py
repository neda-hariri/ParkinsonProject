from tsfresh.utilities.dataframe_functions import impute

import json
import pandas as pd
import matplotlib.pyplot as plt
import math
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

class ThesisPark:
    def starter(self):
        self.calculation()
        #self.calculation_test()

    def calculation_test(self):
        timeseries = pd.read_csv(r'json/test.csv')
        timeseries=timeseries.infer_objects()
        print(timeseries.head())
        extracted_features = extract_features(timeseries, column_id="id", column_sort="time")



    def calculate_distance(self,point1, point2):
        (x1, y1, z1) = point1
        (x2, y2, z2) = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance

    def calculation(self):
        file_paths = ["json/HEALTHY/TRIAL1/FT_30_prova1.json"]
        data_list = []
        for file_path in file_paths:
            with open(file_path) as file:
                data = json.load(file)
                data_list.append(data)

        # Extract the relevant data from the JSON structure
        frames = data['frames']

        # Extract the data for each hand into a list of dictionaries
        hands_data = []
        timestamp_data=[]
        for frame in frames:
            frame_data = frame['hands']
            timestamp = frame['timestamp_usec']
            hands_data.extend(frame_data)
            timestamp_data.append(timestamp)

        # Create a DataFrame from the hands data
        df = pd.DataFrame(hands_data)

        # Extract the specific columns of interest
        df = df[['joints_position']]
        df['timestamp'] = timestamp_data
        # Perform the required computations on the columns
        df['joints_position'] = df['joints_position'].apply(lambda x: self.calculate_distance(x[8], x[4])).astype(float)

        df = df.rename(columns={'joints_position': 'Distance'})

        print(df.head(100))

        diff_df=df.diff().dropna()

        fig, (ax1,ax2)=plt.subplots(1,2,figsize=(12,6))

        ax1.plot(df['Distance'].to_frame())
        #plt.xlabel('Sample')
        #plt.ylabel('Distance in mm')
        #plt.title('Parkinson')

        df['Velocity'] = diff_df['Distance'] / diff_df['timestamp']
        df=df.fillna(df['Velocity'][1])
        ax2.plot(df['Velocity'])
        plt.show()

        # Set id column name and column containing the target values
        #settings = {'id': 'None', 'target': 'Distance'}

        # Extract features
        #extracted_features = extract_features(df, column_id='timestamp', column_sort='timestamp')
       # columns=[(col, col) for col in df.columns]

        #df.columns = pd.MultiIndex.from_tuples(columns)
        #indexList=list(range(0, len(df)))
        #a = df.index.get_level_values(0).astype(float)

        df.index = df.index.get_level_values(0).astype(object)

        extracted_features = extract_features(df, column_id="timestamp", column_sort="Distance")
        #imputed_features = impute(extracted_features)

        #mean_feature = imputed_features['value__mean']
        print(extracted_features)

        # Set id column name and column containing the target values
        #settings = {'id': None, 'target': 'Distance'}

        # Extract features
        #tsfresh.feature_extraction.feature_calculators.absolute_maximum('Distance')
