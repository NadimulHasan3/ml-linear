import pandas as pd
import joblib
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

df  = pd.read_csv('kc_house_data.csv')
pd.set_option('display.max_columns', None)
df = df.filter(['price', 'bedrooms','yr_built', 'bathrooms', 'sqft_living','floors', 'sqft_basement'])
model = lm.LinearRegression()
x= df[[ 'bedrooms', 'bathrooms', 'sqft_living','floors', 'sqft_basement']]
y =df['price']
model.fit(x,y)
prediction =model.predict([[10,5,3000,1,100]]).round()
print(prediction)


#joblib.dump(model, 'housePrice.joblib')

