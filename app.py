from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load your pre-trained model and dataset
with open('decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
df = pd.read_csv('ground water quality.csv')

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

            return render_template('index.html', prediction=prediction, state_options=get_dropdown_options('STATE'), district_options=[], block_options=[], location_options=[])
        
        except IndexError:
            # Handle the case where the row is not found
            return render_template('index.html', error="Data not found. Please check your input.", state_options=get_dropdown_options('STATE'), district_options=[], block_options=[], location_options=[])

    return render_template('index.html', prediction=None, state_options=get_dropdown_options('STATE'), district_options=[], block_options=[], location_options=[])

def get_dropdown_options(column, state=None, district=None, block=None):
    # Get unique values for dropdown options
    if column == 'BLOCK':
        # Filter blocks based on selected state and district
        options = df[(df['STATE'] == state) & (df['DISTRICT'] == district)][column].unique().tolist()
    elif column == 'LOCATION':
        # Filter locations based on selected state, district, and block
        options = df[(df['STATE'] == state) & (df['DISTRICT'] == district) & (df['BLOCK'] == block)][column].unique().tolist()
    else:
        options = df[column].unique().tolist()
    return options

@app.route('/get_districts', methods=['GET'])
def get_districts():
    state = request.args.get('state')
    districts = df[df['STATE'] == state]['DISTRICT'].unique().tolist()
    return jsonify(districts)

@app.route('/get_blocks', methods=['GET'])
def get_blocks():
    state = request.args.get('state')
    district = request.args.get('district')
    blocks = df[(df['STATE'] == state) & (df['DISTRICT'] == district)]['BLOCK'].unique().tolist()
    return jsonify(blocks)

@app.route('/get_locations', methods=['GET'])
def get_locations():
    state = request.args.get('state')
    district = request.args.get('district')
    block = request.args.get('block')
    locations = df[(df['STATE'] == state) & (df['DISTRICT'] == district) & (df['BLOCK'] == block)]['LOCATION'].unique().tolist()
    return jsonify(locations)

if __name__ == '__main__':
    app.run(debug=True)
