# Importing basic libraries
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import configparser
from ml_pipeline.utils import read_data, merge_dataset, interactions
from ml_pipeline.preprocessing import unique_users, unique_items, features_to_add, mapping
from ml_pipeline.model import hybrid_model, evaluate_model
from ml_pipeline.train_test_merge import train_test_merge
from ml_pipeline.recommendation import get_recommendations
from scipy.sparse import csr_matrix

# Read configuration from config.ini file
config = configparser.RawConfigParser()
config.read('/Users/sharmistashastry/Desktop/Self Projects/ML Projects/Hybrid Recommender System/input/config.ini')
DATA_DIR = config.get('DATA', 'data_dir')

# Read the three different datasets we have
order = read_data(DATA_DIR, 'order')
customer = read_data(DATA_DIR, 'customer')
product = read_data(DATA_DIR, 'product')

# Merge the datasets together
full_table = merge_dataset(order, customer, 'CustomerID', 'CustomerID', 'left')
full_table = merge_dataset(full_table, product, 'StockCode', 'StockCode', 'left')

# Creating user, item and feature lists
users = unique_users(order, 'CustomerID')
items = unique_items(product, 'Product Name')
features = features_to_add(customer, 'Customer Segment', 'Age', 'Gender')

# Create mappings from user, items and features to indices and vice-versa as required by LightFM
user_to_index_mapping, index_to_user_mapping, item_to_index_mapping, index_to_item_mapping, features_to_index_mapping, index_to_features_mapping = mapping(users, items, features)

# Focus on data required to create the interaction matrix
user_to_product_rating_train = full_table[['CustomerID', 'Product Name', 'Quantity']]
user_to_product_rating_train = user_to_product_rating_train.groupby(['CustomerID', 'Product Name']).agg({'Quantity': 'sum'}).reset_index()

product_to_feature = full_table[['Product Name', 'Customer Segment', 'Quantity']]
product_to_feature = product_to_feature.groupby(['Product Name', 'Customer Segment']).agg({'Quantity': 'sum'}).reset_index()

# Train-Test split
user_to_product_rating_train, user_to_product_rating_test = train_test_split(user_to_product_rating_train, test_size=0.3, random_state=42)

# Generate user-item interaction matrix
user_to_product_interaction_train = interactions(user_to_product_rating_train, 'CustomerID', 'Product Name', 'Quantity', user_to_index_mapping, item_to_index_mapping)
product_to_feature_interaction = interactions(product_to_feature, 'Product Name', 'Customer Segment', 'Quantity', item_to_index_mapping, features_to_index_mapping)
user_to_product_interaction_test = interactions(user_to_product_rating_test, 'CustomerID', 'Product Name', 'Quantity', user_to_index_mapping, item_to_index_mapping)

# Assuming product_to_feature_interaction is defined and used during training
print("Shape of product_to_feature_interaction during training:", product_to_feature_interaction.shape)

# Build the model, train and evaluate it
model = hybrid_model("logistic", user_to_product_interaction_train, product_to_feature_interaction)

# Evaluate the model
# evaluate = evaluate_model(model, user_to_product_interaction_test, user_to_product_interaction_train, product_to_feature_interaction)

# Merge the train and test data for final model building
user_to_product_interaction = train_test_merge(user_to_product_interaction_train,
                                             user_to_product_interaction_test)

# Convert to csr_matrix
user_to_product_interaction_csr = csr_matrix(user_to_product_interaction)

## Build the final model ##
final_model = hybrid_model("logistic", user_to_product_interaction_csr, product_to_feature_interaction)

## Save the model ##
pickle.dump(final_model, open('../output/final_model.pkl', 'wb'))

## Get the recommendations ##
recommendation_1 = get_recommendations(final_model, 17017, items, user_to_product_interaction, user_to_index_mapping, product_to_feature_interaction)
print(recommendation_1)