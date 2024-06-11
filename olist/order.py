import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True) -> pd.DataFrame:
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        # Hint: Within this instance method, you have access to the instance of the class Order in the variable self, as well as all its attributes

        def calc_delay_vs_expected(delay):
            if delay > pd.to_timedelta(0):
                return delay.days
            return 0

        tmp = (
                self.data["orders"]
                    .loc[lambda _df: _df["order_status"] == "delivered"]
                    .assign(
                        order_purchase_timestamp = lambda _df: pd.to_datetime(_df["order_purchase_timestamp"]),
                        order_approved_at = lambda _df: pd.to_datetime(_df["order_approved_at"]),
                        order_delivered_carrier_date = lambda _df: pd.to_datetime(_df["order_delivered_carrier_date"]),
                        order_delivered_customer_date = lambda _df: pd.to_datetime(_df["order_delivered_customer_date"]),
                        order_estimated_delivery_date = lambda _df: pd.to_datetime(_df["order_estimated_delivery_date"])
                    )
                    .assign(
                        wait_time = lambda _df: (
                            (_df["order_delivered_customer_date"] - _df["order_purchase_timestamp"]).dt.days)
                                .astype("float"),
                        expected_wait_time = lambda _df: (
                            (_df["order_estimated_delivery_date"] - _df["order_purchase_timestamp"]).dt.days)
                                .astype("float"),
                        delay_vs_expected = lambda _df: (_df["order_delivered_customer_date"] - _df["order_estimated_delivery_date"])
                            .apply(calc_delay_vs_expected)
                            .astype("float")
                    )
                    .loc[:, ["order_id", "wait_time", "expected_wait_time", "delay_vs_expected", "order_status"]]
                )
        if is_delivered:
            return tmp.loc[tmp["order_status"] == "delivered"]
        return tmp

    def get_review_score(self) -> pd.DataFrame:
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        return (self.data["order_reviews"]
            .assign(
                dim_is_five_star = lambda _df: (_df["review_score"] == 5).astype("int8"),
                dim_is_one_star = lambda _df: (_df["review_score"] == 1).astype("int8"),
            )
            .loc[:, ["order_id", "dim_is_five_star", "dim_is_one_star", "review_score"]]
        )

    def get_number_products(self):
        """
        Returns a DataFrame with:
        order_id, number_of_products
        """
        return (
            self.data["order_items"]
                .groupby("order_id")["order_item_id"].sum()
                .reset_index(name="number_of_products")
        )

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        return (
            self.data["order_items"]
                .groupby("order_id")["seller_id"].nunique()
                .reset_index(name="number_of_sellers")
        )

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        return (
            self.data["order_items"]
                .groupby("order_id")[["price", "freight_value"]].sum()
                .reset_index()
        )

    # Optional
    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        data = self.data
        sellers = data["sellers"].drop(columns=["seller_city", "seller_state"])
        customers = data["customers"].drop(columns=["customer_unique_id", "customer_city", "customer_state"])
        geolocation = data["geolocation"].drop(columns=["geolocation_city", "geolocation_state"]).groupby("geolocation_zip_code_prefix", as_index=False).mean()

        sellers_geo = (sellers
                        .merge(geolocation, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix")
                        .drop(columns=["geolocation_zip_code_prefix", "seller_zip_code_prefix"])
                        .rename(columns={"geolocation_lat": "geolocation_lat_seller", "geolocation_lng": "geolocation_lng_seller"})
                    )
        customers_geo = (customers
                            .merge(geolocation, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix")
                            .drop(columns=["geolocation_zip_code_prefix", "customer_zip_code_prefix"])
                            .rename(columns={"geolocation_lat": "geolocation_lat_customer", "geolocation_lng": "geolocation_lng_customer"})
                        )

        order_items_sellers = (data["order_items"]
                                .merge(sellers_geo, on="seller_id")
                                .loc[:, ["order_id", "geolocation_lat_seller", "geolocation_lng_seller"]]
                                .drop_duplicates()
                                )



        orders = data["orders"].loc[:, ["order_id", "customer_id"]]

        orders_customers = orders.merge(customers_geo, on="customer_id").drop(columns="customer_id")
        order_customers_sellers = orders_customers.merge(order_items_sellers, on="order_id")

        return (order_customers_sellers
                    .assign(distance_seller_customer =
                        np.vectorize(haversine_distance)(order_customers_sellers.iloc[:, 2],
                                                        order_customers_sellers.iloc[:, 1],
                                                        order_customers_sellers.iloc[:, 4],
                                                        order_customers_sellers.iloc[:, 3])
                    )
                ).loc[:, ["order_id", "distance_seller_customer"]]

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_products', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        # Hint: make sure to re-use your instance methods defined above

        tmp = (
            self.get_wait_time(is_delivered)
                .merge(self.get_review_score(), on="order_id", how="left")
                .merge(self.get_number_products(), on="order_id", how="left")
                .merge(self.get_number_sellers(), on="order_id", how="left")
                .merge(self.get_price_and_freight(), on="order_id", how="left")
        )

        if with_distance_seller_customer:
            return (tmp
                    .merge(self.get_distance_seller_customer(), on="order_id")
                    .dropna()
                )
        return tmp.dropna()
