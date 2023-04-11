from flask import Flask, request, jsonify,render_template
import pickle
from surprise import BaselineOnly
from surprise import Dataset
from surprise.reader import Reader
import pandas as pd
import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--my_variable', type=str, default='default_value')
# args = parser.parse_args()


# Loading the Data
app=Flask(__name__)
@app.route('/recommend')
def recommend_ui():
    return render_template('form.html')
# app.config['MY_VARIABLE'] = args.my_variable
@app.route('/recommend_items',methods=['POST'])
def recommendation():
    # my_variable = app.config['MY_VARIABLE']
    
    customerid = request.form.get('customerid')
    categroy = request.form.get('category')
    df_merged=pd.read_csv('df_merged.csv')
    final_df=pd.read_csv('final_df.csv')
    df=final_df.iloc[0:2000]

    reader = Reader()
    ratings = Dataset.load_from_df(df[['customerid', 'id', 'avgrating']], reader)

    # Load pre-trained model
    with open('baseline_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Set up Flask app


        # Define function to return recommendations

        # Get recommendations for specified user ID
    testset = ratings.build_full_trainset().build_anti_testset()
    predictions = model.test(testset)

    def get_top_n_recommendations(customerid,predictions, n=20):
        predict_ratings = {}
        # loop for getting predictions for the user
        for uid, iid, true_r, est, _ in predictions:
            if (uid==customerid):
                predict_ratings[iid] = est
        predict_ratings = sorted(predict_ratings.items(), key=lambda kv: kv[1],reverse=True)[:n]
        top_items = [i[0] for i in predict_ratings]
        # print(top_items)
        # top_items = [str(i) for i in top_items]
        # print("="*10,"Recommended movies for user {} :".format(customerid),"="*10)
        # print(df_merged[df_merged["id"].isin(top_items)][["item_name","market","price","avgrating"]].to_string(index=False))
        top_item= df_merged[df_merged["id"].isin(top_items)][["item_name","market","price","avgrating"]].sort_values(by=['avgrating'],ascending=False)
        return top_item
    top_n = get_top_n_recommendations(customerid,predictions)

    g1=top_n.groupby('market')

    def best_pick(item):
        df_item=g1.get_group(item)
        final_item=df_item[['avgrating','item_name','price',]]
        best_pick_item=final_item.sort_values(by='avgrating',ascending=False)
        best_pick_item.drop_duplicates('item_name',inplace=True)
        return best_pick_item.head()
    
    best_items=best_pick(categroy)
    data_dict = best_items.to_dict()
    return data_dict
    #print(data_dict)



if __name__=="__main__":
    app.run(port=2022, debug=True)