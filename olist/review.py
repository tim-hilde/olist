import pandas as pd
import numpy as np
import math
from olist.data import Olist
from olist.order import Order


class Review:

    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_review_length(self):
        """
        Returns a DataFrame with:
       'review_id', 'length_review', 'review_score'
        """
        pass  # YOUR CODE HERE

    def get_main_product_category(self):
        """
        Returns a DataFrame with:
       'review_id', 'order_id','product_category_name'
        """
        pass  # YOUR CODE HERE

    def get_training_data(self):
        pass  # YOUR CODE HERE
