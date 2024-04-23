import pandas as pd
import numpy as np
from math import sqrt
from olist.data import Olist
from olist.order import Order


class Seller:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers'].copy(
        )  # Make a copy before using inplace=True so as to avoid modifying self.data
        sellers.drop('seller_zip_code_prefix', axis=1, inplace=True)
        sellers.drop_duplicates(
            inplace=True)  # There can be multiple rows per seller
        return sellers

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
        'seller_id', 'delay_to_carrier', 'wait_time'
        """
        # Get data
        order_items = self.data['order_items'].copy()
        orders = self.data['orders'].query("order_status=='delivered'").copy()

        ship = order_items.merge(orders, on='order_id')

        # Handle datetime
        ship.loc[:, 'shipping_limit_date'] = pd.to_datetime(
            ship['shipping_limit_date'])
        ship.loc[:, 'order_delivered_carrier_date'] = pd.to_datetime(
            ship['order_delivered_carrier_date'])
        ship.loc[:, 'order_delivered_customer_date'] = pd.to_datetime(
            ship['order_delivered_customer_date'])
        ship.loc[:, 'order_purchase_timestamp'] = pd.to_datetime(
            ship['order_purchase_timestamp'])

        # Compute delay and wait_time
        def delay_to_logistic_partner(d):
            days = np.mean(
                (d.order_delivered_carrier_date - d.shipping_limit_date) /
                np.timedelta64(24, 'h'))
            if days > 0:
                return days
            else:
                return 0

        def order_wait_time(d):
            days = np.mean(
                (d.order_delivered_customer_date - d.order_purchase_timestamp)
                / np.timedelta64(24, 'h'))
            return days

        delay = ship.groupby('seller_id')\
                    .apply(delay_to_logistic_partner)\
                    .reset_index()
        delay.columns = ['seller_id', 'delay_to_carrier']

        wait = ship.groupby('seller_id')\
                   .apply(order_wait_time)\
                   .reset_index()
        wait.columns = ['seller_id', 'wait_time']

        df = delay.merge(wait, on='seller_id')

        return df

    def get_active_dates(self):
        """
        Returns a DataFrame with:
        'seller_id', 'date_first_sale', 'date_last_sale', 'months_on_olist'
        """
        # First, get only orders that are approved
        orders_approved = self.data['orders'][[
            'order_id', 'order_approved_at'
        ]].dropna()

        # Then, create a (orders <> sellers) join table because a seller can appear multiple times in the same order
        orders_sellers = orders_approved.merge(self.data['order_items'],
                                               on='order_id')[[
                                                   'order_id', 'seller_id',
                                                   'order_approved_at'
                                               ]].drop_duplicates()
        orders_sellers["order_approved_at"] = pd.to_datetime(
            orders_sellers["order_approved_at"])

        # Compute dates
        orders_sellers["date_first_sale"] = orders_sellers["order_approved_at"]
        orders_sellers["date_last_sale"] = orders_sellers["order_approved_at"]
        df = orders_sellers.groupby('seller_id').agg({
            "date_first_sale": min,
            "date_last_sale": max
        })
        df['months_on_olist'] = round(
            (df['date_last_sale'] - df['date_first_sale']) /
            np.timedelta64(1, 'M'))
        return df

    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity', 'quantity_per_order'
        """
        order_items = self.data['order_items']

        n_orders = order_items.groupby('seller_id')['order_id']\
            .nunique()\
            .reset_index()
        n_orders.columns = ['seller_id', 'n_orders']

        quantity = order_items.groupby('seller_id', as_index=False).agg(
            {'order_id': 'count'})
        quantity.columns = ['seller_id', 'quantity']

        result = n_orders.merge(quantity, on='seller_id')
        result['quantity_per_order'] = result['quantity'] / result['n_orders']
        return result

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        return self.data['order_items'][['seller_id', 'price']]\
            .groupby('seller_id')\
            .sum()\
            .rename(columns={'price': 'sales'})

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score'
        """

        # $CHALLENGIFY_BEGIN
        orders_reviews = self.order.get_review_score()
        orders_sellers = self.data['order_items'][['order_id', 'seller_id'
                                                   ]].drop_duplicates()

        df = orders_sellers.merge(orders_reviews, on='order_id')
        res = df.groupby('seller_id', as_index=False).agg({
            'dim_is_one_star':
            'sum',
            'dim_is_two_star':
            'sum',
            'dim_is_three_star':
            'sum',
            'dim_is_four_star':
            'sum',
            'dim_is_five_star':
            'sum',
            'review_score':
            'mean'
        })
        # Rename columns
        res.columns = [
            'seller_id', 'sum_of_one_stars', 'sum_of_two_stars','sum_of_three_stars',\
                'sum_of_four_stars','sum_of_five_stars','review_score'
        ]

        return res
        # $CHALLENGIFY_END


    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['seller_id', 'seller_city', 'seller_state', 'delay_to_carrier',
        'wait_time', 'date_first_sale', 'date_last_sale', 'months_on_olist', 'share_of_one_stars',
        'share_of_five_stars', 'review_score', 'n_orders', 'quantity',
        'quantity_per_order', 'sales']
        """

        training_set =\
            self.get_seller_features()\
                .merge(
                self.get_seller_delay_wait_time(), on='seller_id'
               ).merge(
                self.get_active_dates(), on='seller_id'
               ).merge(
                self.get_quantity(), on='seller_id'
               ).merge(
                self.get_sales(), on='seller_id'
               )

        if self.get_review_score() is not None:
            training_set = training_set.merge(self.get_review_score(),
                                              on='seller_id')


        review_score_cost = {'1 star': 100,'2 stars': 50,'3 stars': 40,'4 stars': 0,'5 stars': 0}

        training_set['revenues']= training_set['sales']*0.1 + training_set['months_on_olist']*80

        training_set['cost_of_reviews'] = training_set['sum_of_one_stars']*review_score_cost['1 star'] + training_set['sum_of_two_stars'] * \
                review_score_cost['2 stars'] + training_set['sum_of_three_stars'] * review_score_cost['3 stars']

        training_set['profits']=training_set['revenues']-training_set['cost_of_reviews']


        return training_set.sort_values(by='profits', ascending=True)


    def get_IT_cost_savings(self, n):
        """Returns the cost savings from removing the worst n numbers of sellers from the platform"""

        self.n = n

        IT_costs_df = self.get_training_data().sort_values(by='profits', ascending=True)

        alpha = 3157.27
        beta = 978.23

        old_n_sellers = len(IT_costs_df)
        old_n_items = IT_costs_df['quantity'].sum()

        new_n_sellers = old_n_sellers - self.n
        new_n_items = IT_costs_df['quantity'].sum() - IT_costs_df.iloc[0:self.n]['quantity'].sum()

        old_IT_costs = alpha * sqrt(old_n_sellers) + beta * sqrt(old_n_items)
        new_IT_costs = alpha * sqrt(new_n_sellers) + beta * sqrt(new_n_items)

        IT_costs_savings = old_IT_costs - new_IT_costs

        return IT_costs_savings


    def get_improved_profits(self, n):
        """Returns the improved profits from removing the worst n numbers of sellers from the platform"""

        self.n = n

        profits_df = self.get_training_data().sort_values(by='profits', ascending=True)

        improved_profits = profits_df.iloc[0:self.n]['profits'].sum()*-1

        return improved_profits


    #def get_max_profit(self, n)
       # df = pd.DataFrame(columns='IT_Savings', 'Profit_Improvement', 'Total_Improvement']

        #for i in range(1, n + 1):  # Start from 1 to n, assuming you want to calculate for each number of sellers removed
      #      savings = seller.get_IT_cost_savings(i)
    #profit_improvement = seller.get_improved_profits(i)
    #total_improvement = savings + profit_improvement

    # Append the calculated values to the DataFrame
    #df = df.append({'Savings': savings, 'Profit_Improvement': profit_improvement, 'Total_Improvement': total_improvement}, ignore_index=True)

# old_profits = profits_df['profits'].sum()
# new_profits = profits_df['profits'].sum() - profits_df.iloc[0:self.n]['profits'].sum()
