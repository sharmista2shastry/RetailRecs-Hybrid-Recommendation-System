# RetailRecs - A Hybrid Recommender System

This project implements a hybrid recommender system that combines collaborative filtering and content-based filtering to provide personalized recommendations.

## Objective

There are two main methods for making these suggestions: content-based and collaborative filtering. Collaborative filtering finds similarities between users to make recommendations, while content-based filtering personalizes content for each user based on their previous actions and feedback.

However, these methods struggle when there's not enough data. To address this, we'll explore a Hybrid Recommendation System, which combines both approaches.

## Data Description

The dataset used in this project contains transactional data for a UK-based online retail company that sells unique gifts for various occasions.

## Approach

1. Import required libraries
2. Read and merge the data
3. Prepare the data
4. Split the data into training and testing sets
5. Build models
6. Model with WARP loss function
7. Model with logistic loss function
8. Model with BPR loss function
9. Combine data for the final model
10. Generate recommendations

## Install all the requirements

pip install -r requirements.txt