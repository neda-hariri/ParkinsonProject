import math
import os
import pandas as pd
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