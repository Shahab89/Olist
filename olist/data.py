import os
import pandas as pd


class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """
        # Hints 1: Build csv_path as "absolute path" in order to call this method from anywhere.
            # Do not hardcode your path as it only works on your machine ('Users/username/code...')
            # Use __file__ instead as an absolute path anchor independant of your usename
            # Make extensive use of `breakpoint()` to investigate what `__file__` variable is really
        # Hint 2: Use os.path library to construct path independent of Mac vs. Unix vs. Windows specificities

        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data','csv')
        print(csv_path)

        file_names = [file for file in os.listdir(csv_path) if file.endswith('.csv')]
        key_names = [file.replace(".csv", "").replace("_dataset", "").replace("olist_", "") for file in file_names]
        data = {}

        for (key,file) in zip(key_names, file_names):
            data[key]= pd.read_csv(os.path.join(csv_path, file))

        return data


    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")


# if __name__ == '__main__':
#     olist = Olist()
#     olist.get_data()
