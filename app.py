from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load your pre-trained model
with open('decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load your dataset
df = pd.read_csv('ground water quality.csv')  # Update with your actual dataset file

# Define a route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        state = request.form['state']
        district = request.form['district']
        block = request.form['block']
        location = request.form['location']

        # Find the corresponding row in the dataset
        try:
            row = df.loc[(df['STATE'] == state) & (df['DISTRICT'] == district) & (df['BLOCK'] == block) & (df['LOCATION'] == location)].iloc[0]
            
            # Extract features for prediction
            features = row[['pH', 'EC', 'Cl', 'SO4', 'NO3', 'TH', 'TDS']].values.reshape(1, -1)
            
            # Make the prediction
            prediction = model.predict(features)[0]

            return render_template('index.html', prediction=prediction)
        
        except IndexError:
            # Handle the case where the row is not found
            return render_template('index.html', error="Data not found. Please check your input.")

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
