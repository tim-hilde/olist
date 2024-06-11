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


        csv_path = __file__.removesuffix("/olist/data.py") + "/data/csv"

        file_names = [file for file in os.listdir(csv_path) if file.endswith(".csv")]

        key_names = [
            file_name.removesuffix(".csv").removesuffix("_dataset").removeprefix("olist_")
            for file_name in file_names
        ]

        data = {
            key_name: pd.read_csv(os.path.join(csv_path, file_name))
            for key_name, file_name in zip(key_names, file_names)
        }

        return data


    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
