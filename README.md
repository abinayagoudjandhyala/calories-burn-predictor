
# Calories Prediction using Machine Learning

## Project Overview
This project predicts the calories burned during exercise using a machine learning regression model.  
Dataset used: [Kaggle Dataset - Exercise and Health](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos).

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-Learn  
- Matplotlib  
- Seaborn  
- Google Colab  

## How to Run the Project on Google Colab
1. Upload the notebook (CaloriesBurntPrediction.ipynb) and dataset files to Google Drive.  
2. Open the notebook in Google Colab.  
3. Install dependencies by running:
```python
!pip install numpy pandas scikit-learn matplotlib seaborn
```
4. Run all cells sequentially to train the model and make predictions.  

## Sample Prediction Code
```python
input_data = (1, 68, 190, 94, 29, 105, 40)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print('The Calories Burned is', prediction[0])
```

## Results
The model provides accurate predictions for calories burned based on input values.  

## License
This project is licensed under the MIT License.


