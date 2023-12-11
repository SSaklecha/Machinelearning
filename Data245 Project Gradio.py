import gradio as gr
import pandas as pd
import pickle

# Load the trained model
pickle_file_path = 'C:/Users/abdul/Desktop/Data245 project ML model.pkl'
with open(pickle_file_path, 'rb') as file:
    stacking_model = pickle.load(file)

selected_columns_ML = [
    'last_fico_range_low',
    'emp_title',
    'total_rec_prncp',
    'loan_amnt',
    'days_since_last_cred_pull',
    'installment',
    'int_rate',
    'total_rec_int',
    'tot_cur_bal',
    'days_since_issue_d',
    'annual_inc',
    'out_prncp',
    'tot_hi_cred_lim',
    'sub_grade',
    'addr_state',
    'total_acc'
]

def predict_default(file):
    # Load the data from the csv file
    data = pd.read_csv(file.name)
    # Make sure the data has the correct columns
    assert set(selected_columns_ML).issubset(set(data.columns)), "Input data must have the correct columns"
    # Select only the columns in selected_columns_ML
    feature_data = data[selected_columns_ML]
    # Make predictions
    predictions = stacking_model.predict(feature_data)
    # Create a DataFrame with the id and the predictions
    result = pd.DataFrame({
        'id': data['id'],
        'prediction': ['default' if p == 0 else 'non-default' for p in predictions]
    })
    # Return the result as a string
    return '\n'.join(result.apply(lambda row: f"id: {row['id']}, prediction: {row['prediction']}", axis=1))

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_default, 
    inputs=gr.File(label="Upload CSV file"),  # Corrected input definition
    outputs="text",
    title="Loan Default Prediction",
    description="Upload a CSV file to predict whether each loan will default or not."
)

# Launch the interface
iface.launch(share=False)