from crypt import methods
from flask import Flask,render_template,request,jsonify
import pickle
import pandas as pd
from datetime import date

print(pd.__version__)

file=pickle.load(open('model/groupByMarket.pkl','rb'))
data_merged=pd.read_csv('df_merged.csv')
app=Flask(__name__)

@app.route('/bestpick_item',methods=['POST'])    
def best_pick():
    #return ("rashi")
    data = request.form['data']
    filtered_data = file.get_group(data)
    # #top_20_items = sorted(filtered_data, key=lambda x: x['avgRating'], reverse=True)[:20]
    # #top_20_items = pd.DataFrame(top_20_items)
    # #top_20_items.drop_duplicates('item_name',inplace=True)
    top_20_items = filtered_data.sort_values(by='avgrating', ascending=False)[:21]
    top_20_items = top_20_items.drop_duplicates('item_name',inplace=True)
    json_data = top_20_items.to_json(orient='records')
    # print(json_data)
    return(json_data)

@app.route('/new_arrival',methods=['GET'])    
def new_arrival():
    # col=data_merged.columns
    # return col
    data_merged.drop('Unnamed: 0',axis=1)
    final_df=data_merged.dropna(subset=['createdat'])
    today1=date.today()
    today1=pd.to_datetime(today1)
    print(type(today1))
    final_df['latest']=today1-(pd.to_datetime(final_df.createdat))
    new_items=final_df[["id","item_name","market","price","avgrating",'id','images','dealprice','type','discount','discountpercent','latest']].sort_values(by=['latest'],ascending=True)[:5]
    new_items['latest'] = new_items['latest'].dt.days.astype('int16')
    dict_data = new_items.to_dict()
    return dict_data
  	
if __name__ == '__main__':
    app.run(debug=True)




