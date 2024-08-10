from lightfm import LightFM
import time
from lightfm.evaluation import auc_score

def hybrid_model(loss, interaction_matrix, product_matrix):
    if loss == "warp":
        model = LightFM(loss="warp")
        start = time.time()

        model.fit_partial(interaction_matrix, user_features=None, item_features=product_matrix, epochs=1, num_threads=4)

        end = time.time()
        print("time taken for fitting = {0:.{1}f} seconds".format(end - start, 2))

        return model
    
    if loss == "logistic":
        model = LightFM(loss="logistic", no_components=30)
        start = time.time()

        model.fit_partial(interaction_matrix, user_features=None, item_features=interaction_matrix, epochs=10, num_threads=20)

        end = time.time()
        print("time taken for fitting = {0:.{1}f} seconds".format(end - start, 2))
        
        return model
    
    if loss == "bpr":
        model = LightFM(loss="bpr")
        start = time.time()

        model.fit_partial(interaction_matrix, user_features=None, item_features=interaction_matrix, epochs=1, num_threads=4)

        end = time.time()
        print("time taken for fitting = {0:.{1}f} seconds".format(end - start, 2))
        
        return model
    
    else:
        print("Invalid loss function specified")

def evaluate_model(model, interaction_test, interaction_train, product_interaction):
    start = time.time()  # Record the start time

    auc_with_features = auc_score(model=model,  
        test_interactions=interaction_test,
        train_interactions=interaction_train, 
        item_features=product_interaction,
        num_threads=4,
        check_intersections=False)  # Calculate AUC score

    end = time.time()  # Record the end time

    print("time taken for AUC score = {0:.{1}f} seconds".format(end - start, 2))

    return "average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2)