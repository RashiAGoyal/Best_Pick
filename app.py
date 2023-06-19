#from crypt import methods
from flask import Flask,render_template,request,jsonify
import pickle
import pandas as pd
from datetime import date
from surprise import Dataset
from surprise.reader import Reader

print(pd.__version__)

chunksize=1000

file=pd.read_pickle(open('model/groupByMarket.pkl','rb'))
final_df=pd.read_pickle(open('model/merge_item_orderitem_address.pkl','rb'))
#unpickler = pickle.Unpickler(final_df)
app=Flask(__name__)

@app.route('/new_arrival',methods=['GET'])    
def new_arrival():
    #final_df.drop('Unnamed: 0',axis=1)
    today1 = date.today()
    today1 = pd.to_datetime(today1)

    # Get the page number from query parameters
    page = int(request.args.get('page', 1))

    # Get the page size from query parameters
    page_size = int(request.args.get('page_size', 20))

    # Calculate the start and end indices based on the page and page size
    start_index = (page - 1) * page_size
    end_index = start_index + page_size

    # Sort the data by 'latest' column in ascending order and select the specified range
    final_df['latest']=today1-final_df.createdat
    new_items = final_df[["id", "item_name", "market", "price", "avgrating", 'id', 'images', 'dealprice', 'type', 'discount', 'discountpercent', 'latest']].sort_values(by=['latest'], ascending=True)[start_index:end_index]
    new_items['latest'] = new_items['latest'].dt.days.astype('int16')
    dict_data = new_items.to_dict()
    return dict_data

@app.route('/bestpick_item',methods=['POST'])    
def best_pick():
    category = request.form['category']
    filtered_data = file.get_group(category)

    # Get the page number from query parameters
    page = int(request.args.get('page', 1))

    # Get the page size from query parameters
    page_size = int(request.args.get('page_size', 20))

    # Calculate the start and end indices based on the page and page size
    start_index = (page - 1) * page_size
    end_index = start_index + page_size

    # Sort the filtered data by 'avgrating' column in descending order and select the specified range
    top_20_items = filtered_data.sort_values(by='avgrating', ascending=False).drop_duplicates('item_name')[start_index:end_index]

    json_data = top_20_items.to_json(orient='records')
    return json_data

def get_location(location):
    # Logic to fetch data based on the location
    # Replace this with your own implementation
    data = {
        'location': location
           }
    return data
def get_customerId(custId):
    # Logic to fetch data based on the location
    # Replace this with your own implementation
    data = {
        'customerId': custId
           }
    return data

@app.route('/trending_item',methods=['GET'])
def trending_items():
    request_loc = request.get_json()
    if request_loc and 'location' in request_loc:
        # Extract the location from the JSON data
        loc = request_loc['location']
        # Call the get_data function with the location argument
        location = get_location(loc)
        today = date.today()
        today=pd.to_datetime(today)
        #final_df = pd.DataFrame()
    #location = get_data(location)
        print(today)
        print(location)    
        final_df['latest']=today-final_df.order_date
        new_items= final_df[["id","item_name","market","price","avgrating",'id','images','dealprice','type','discount','discountpercent','latest','state','qty']].sort_values(by=['latest'],ascending=True)
        df_final=new_items.loc[(new_items.latest<='100 days')]
        num_qty_df = df_final.groupby('item_name').count()['qty'].reset_index()
        num_qty_df.rename(columns={'qty':'num_qty'},inplace=True)
        data_final=df_final.merge(num_qty_df,on='item_name')
        g=data_final.groupby(['state'])
        data_state=g.get_group(loc)
        final_item=data_state[['avgrating','item_name','price','num_qty']]
        trending_item=final_item.sort_values(by='num_qty',ascending=False)
        trending_item.drop_duplicates('item_name',inplace=True)
    #trending_dict=trending_item.to_dict()
        return jsonify(trending_item.to_dict(orient='records'))  
    return jsonify({'error': 'Invalid request. Please provide a location.'}), 400

@app.route('/suggested_for_you',methods=['GET'])    
def recommendation():    
    #final_df=final_df.dropna(subset=['customerid'])
    reader = Reader()
    with open('model/svd_model.pkl', 'rb') as f:
        model = pickle.load(f)
    predictions=[]
    #for chunk in range(unpickler):
    ratings = Dataset.load_from_df(final_df[['customerid', 'id', 'avgrating']], reader)
    testset = ratings.build_full_trainset().build_anti_testset()
    prediction = model.test(testset)
    predictions.extend(prediction)
    
    def get_top_n_recommendations(predictions, n=10000):
        predict_ratings = {}
        request_id = request.get_json()
        if request_id and 'customerId' in request_id:
        # Extract the customerId from the JSON data
            custId = request_id['customerId']
            custId = get_customerId(custId)
            # loop for getting predictions for the user
            for uid, iid, true_r, est, _ in predictions:
                if (uid==custId):
                    predict_ratings[iid] = est
            predict_ratings = sorted(predict_ratings.items(), key=lambda kv: kv[1],reverse=True)[:n]
            top_items = [i[0] for i in predict_ratings]
            top_item= final_df[final_df["id"].isin(top_items)][["item_name","market","price","avgrating",'id','images','dealprice','type','discount','discountpercent','state']].sort_values(by=['avgrating'],ascending=False).head(5)
            return top_item
    top_n = get_top_n_recommendations(predictions)
    top_n_dict = top_n.to_dict()
    return top_n_dict

    
if __name__ == '__main__':
    app.run(debug=True)




