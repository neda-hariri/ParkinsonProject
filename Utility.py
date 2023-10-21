import math
import os
import pandas as pd
from Configs import Configs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Utility:
    def get_dataframe_from_excel(self, uri):
        df = pd.read_excel(uri, sheet_name=None)
        return df

    @staticmethod
    def calculate_distance(point1, point2):
        (x1, y1, z1) = point1
        (x2, y2, z2) = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance

    def get_files_in_directory(self, directory):
        file_list = []
        for root, directories, files in os.walk(directory):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    def get_test_train_from_dataframe(self, data, labels):
        return train_test_split(data, labels, test_size=0.25, random_state=80)

    def get_configs(self, is_pd):
        if is_pd:
            file_paths = self.get_files_in_directory("json/PD/")
            jointPositionTags = 'joint_positions'
            output_extension = '_parkinson'
        else:
            file_paths = self.get_files_in_directory("json/HEALTHY/")
            jointPositionTags = 'joints_position'
            output_extension = '_healthy'

        frames = 'frames'
        hands = 'hands'
        timestamp_usec = 'timestamp_usec'
        timestamp = 'timestamp'
        distance = 'Distance'
        velocity = 'Velocity'
        img_output_dir = 'Imgs'
        is_pd = "is_pd"
        return Configs(file_paths, jointPositionTags, frames, hands, timestamp_usec, timestamp, distance, velocity,
                       output_extension, img_output_dir, is_pd)

    def save_graphs(self, df, configs, json_file_path):
        json_file_name = os.path.basename(json_file_path)

        # Create folders if they don't exist
        modified_uri = json_file_path.replace('\\', '/')
        modified_uri = modified_uri.replace('.json', '')

        distance_dir = os.path.join(configs.img_output_dir, modified_uri, configs.distance)
        velocity_dir = os.path.join(configs.img_output_dir, modified_uri, configs.velocity)
        distance_dir = distance_dir.replace('\\', '/')
        velocity_dir = velocity_dir.replace('\\', '/')
        os.makedirs(distance_dir, exist_ok=True)
        os.makedirs(velocity_dir, exist_ok=True)

        # Plot and save line chart for distance
        distance_chart = plt.figure()
        plt.plot(df[configs.timestamp], df[configs.distance])
        plt.xlabel('Timestamp')
        plt.ylabel('Distance')
        plt.title('Distance vs Timestamp')
        distance_chart_path = os.path.join(distance_dir, 'distance_chart' + json_file_name + '.png')
        plt.savefig(distance_chart_path)
        plt.close(distance_chart)

        # Plot and save line chart for velocity
        velocity_chart = plt.figure()
        plt.plot(df[configs.timestamp], df[configs.velocity])
        plt.xlabel('Timestamp')
        plt.ylabel('Velocity')
        plt.title('Velocity vs Timestamp')
        velocity_chart_path = os.path.join(velocity_dir, 'velocity_chart' + json_file_name + '.png')
        plt.savefig(velocity_chart_path)
        plt.close(velocity_chart)