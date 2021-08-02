from pickle import load
model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
print(model.predict(scaler.transform([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])))