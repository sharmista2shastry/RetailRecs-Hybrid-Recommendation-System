import numpy as np
from scipy.sparse import csr_matrix

def get_recommendations(model, user, items, user_to_product_interaction_matrix, user2index_map, product_to_feature_interaction_matrix):
    # Get the user's index, return None if user doesn't exist
    user_index = user2index_map[user]
    if user_index is None:
        return None
    
    if not isinstance(user_to_product_interaction_matrix, csr_matrix):
        user_to_product_interaction_matrix = csr_matrix(user_to_product_interaction_matrix)
    
    # Get products already bought by the user
    known_positives = items[user_to_product_interaction_matrix[user_index].indices]
    
    # Predict scores using the model
    scores = model.predict(user_ids=user_index, 
                           item_ids=np.arange(user_to_product_interaction_matrix.shape[1]), 
                           item_features=product_to_feature_interaction_matrix)
    
    # Get top recommended items based on scores
    top_items = items[np.argsort(-scores)]
    
    # Print results
    print(f"User {user}")
    print("     Known positives:")
    for x in known_positives[:10]:
        print(f"                  {x}")
    
    print("     Recommended:")
    for x in top_items[:10]:
        print(f"                  {x}")

