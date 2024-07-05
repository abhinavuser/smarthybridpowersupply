import csv


# Data for the dataset
data = [
    ['Date', 'Temperature (°C)', 'Humidity (%)', 'Precipitation (mm)', 'Wind Speed (km/h)', 'Weather Condition'],
    ['2024-02-01', 30, 70, 0, 10, 'Sunny'],
    ['2024-02-02', 29, 75, 0.5, 12, 'Partly Cloudy'],
    ['2024-02-03', 31, 68, 0, 15, 'Sunny'],
    ['2024-02-04', 32, 72, 1, 14, 'Light Rain'],
    ['2024-02-05', 28, 80, 0, 11, 'Cloudy']
]

# Saving data to a CSV file
with open('chennai_weather_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)

print("Dataset created and saved successfully!")

# Load the dataset
with open('chennai_weather_data.csv', 'r') as file:
    data = [line.strip().split(',') for line in file.readlines()]

# Analyze energy consumption patterns
energy_demand = []
for row in data[1:]:
    try:
        demand = float(row[-1])
        energy_demand.append(demand)
    except ValueError:
        continue

if energy_demand:
    min_demand = min(energy_demand)
    max_demand = max(energy_demand)
    average_demand = sum(energy_demand) / len(energy_demand)
else:
    min_demand = max_demand = average_demand = None


#Correlation analysis (example: calculate correlation between temperature and energy demand)
temperature = []
for row in data[1:]:
    try:
        temp = float(row[1])
        temperature.append(temp)
    except ValueError:
        continue

def correlation_coefficient(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum(xi * xi for xi in x)
    sum_y_sq = sum(yi * yi for yi in y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))**0.5
    
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

correlation_coefficient = correlation_coefficient(temperature, energy_demand)

# Adding time features
import datetime
for row in data[1:]:
    date = datetime.datetime.strptime(row[0], '%Y-%m-%d')
    row.append(date.hour)
    row.append(date.weekday())

# Engineer additional features (example: interaction between temperature and humidity)
for row in data[1:]:
    temp_humidity_interaction = float(row[1]) * float(row[2])
    row.append(temp_humidity_interaction)

# Splitting the dataset
def train_test_split(data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]

# Training a linear regression model
def linear_regression_fit(X_train, y_train):
    n = len(X_train)
    num_features = len(X_train[0])
    
    # Initialize sums for each feature
    sum_x = [0.0] * num_features
    sum_y = sum(y_train)
    sum_x_sq = [0.0] * num_features
    sum_xy = [0.0] * num_features
    
    # Calculate sums for each feature
    for x_values, y in zip(X_train, y_train):
        try:
            x_values = [float(x) for x in x_values]  # Convert values to floats
        except ValueError:
            continue  # Skip row if conversion fails
        
        for i, x in enumerate(x_values):
            sum_x[i] += x
            sum_x_sq[i] += x ** 2
            sum_xy[i] += x * y
    
    # Calculate coefficients (slope and intercept) for each feature
    slopes = []
    for i in range(num_features):
        if n * sum_x_sq[i] - sum_x[i] ** 2 != 0:
            slope_i = (n * sum_xy[i] - sum_x[i] * sum_y) / (n * sum_x_sq[i] - sum_x[i] ** 2)
        else:
            slope_i = 0  # Set slope to 0 if denominator is zero
        slopes.append(slope_i)
    
    intercept = (sum_y - sum(slopes[i] * sum_x[i] for i in range(num_features))) / n
    
    return slopes, intercept


# Making predictions
def linear_regression_predict(X_test, slopes, intercept):
    predictions = []
    for x_values in X_test:
        try:
            x_values = [float(x) for x in x_values]  # Convert values to floats
        except ValueError:
            predictions.append(float('nan'))  # Add NaN if conversion fails
            continue
         
        # Calculate predicted value using linear regression equation: y = m1*x1 + m2*x2 + ... + b
        prediction = sum(slope * x + intercept for slope, x in zip(slopes, x_values))
        predictions.append(prediction)
    
    return predictions


# Splitting the dataset into features and target variable
X = [row[:-1] for row in data[1:]]
y = [float(row[-1]) for row in data[1:]]

# Splitting the dataset into training and testing sets
X_train, X_test = train_test_split(X)
y_train, y_test = train_test_split(y)

# Training a linear regression model
slope, intercept = linear_regression_fit(X_train, y_train)

# Making predictions
predictions = linear_regression_predict(X_test, slope, intercept)

# Evaluating the model (mean squared error)
mse = sum((true - pred)**2 for true, pred in zip(y_test, predictions)) / len(y_test)
print("Mean Squared Error:", mse)

# Define a node for decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold          # Threshold value for the split
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        self.value = value                  # Value if node is a leaf
import random

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            sample_indices = random.sample(range(len(X)), len(X))
            X_sampled = [X[i] for i in sample_indices]
            y_sampled = [y[i] for i in sample_indices]
            tree.fit(X_sampled, y_sampled)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for x in X:
            tree_predictions = [tree.predict([x])[0] for tree in self.trees]
            predictions.append(sum(tree_predictions) / len(tree_predictions))
        return predictions

# Define the rest of the code for grid search and evaluation

# Define a decision tree regressor
class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):

        n_samples, n_features = len(X), len(X[0])
        variance = self._variance(y)
        best_variance_reduction = 0
        best_split = None

        for feature_index in range(n_features):
            values = [X[i][feature_index] for i in range(n_samples) if isinstance(X[i][feature_index], (int, float))]
            if not values:
                continue  # Skip feature if all values are non-numeric

            thresholds = sorted(set(values))
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2.0
                left_indices = [j for j in range(n_samples) if X[j][feature_index] <= threshold]
                right_indices = [j for j in range(n_samples) if X[j][feature_index] > threshold]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_variance = self._variance([y[j] for j in left_indices])
                right_variance = self._variance([y[j] for j in right_indices])
                weighted_variance = (len(left_indices) / n_samples) * left_variance + (len(right_indices) / n_samples) * right_variance
                variance_reduction = variance - weighted_variance

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }

        if best_variance_reduction > 0 and (self.max_depth is None or depth < self.max_depth):
            left = self._build_tree([X[i] for i in best_split['left_indices']], [y[i] for i in best_split['left_indices']], depth + 1)
            right = self._build_tree([X[i] for i in best_split['right_indices']], [y[i] for i in best_split['right_indices']], depth + 1)
            return Node(feature_index=best_split['feature_index'], threshold=best_split['threshold'], left=left, right=right)
        else:
            return Node(value=sum(y) / len(y))



    def _variance(self, arr):
        mean = sum(arr) / len(arr)
        return sum((x - mean) ** 2 for x in arr) / len(arr)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]


# Hyperparameter tuning example
def grid_search(X_train, y_train):
    best_mse = float('inf')
    best_model = None
    
    for n_estimators in [50, 100, 150]:
        for max_depth in [None, 10, 20]:
            # Train a model
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate MSE
            mse = sum((true - pred)**2 for true, pred in zip(y_test, predictions)) / len(y_test)
            
            # Update best model
            if mse < best_mse:
                best_mse = mse
                best_model = model
    
    return best_model

# Splitting the dataset into features and target variable
X = [row[:-1] for row in data[1:]]
y = [float(row[-1]) for row in data[1:]]

# Splitting the dataset into training and testing sets
X_train, X_test = train_test_split(X)
y_train, y_test = train_test_split(y)

# Hyperparameter tuning
best_model = grid_search(X_train, y_train)

# Training the best model
best_model.fit(X_train, y_train)

# Making predictions
predictions = best_model.predict(X_test)

# Evaluating the model (mean squared error)
mse = sum((true - pred)**2 for true, pred in zip(y_test, predictions)) / len(y_test)
print("Mean Squared Error:", mse)


# Splitting the dataset into features and target variable
X = [row[:-1] for row in data[1:]]
y = [float(row[-1]) for row in data[1:]]

# Splitting the dataset into training and testing sets
X_train, X_test = train_test_split(X)
y_train, y_test = train_test_split(y)

# Hyperparameter tuning
best_model = grid_search(X_train, y_train)

# Training the best model
best_model.fit(X_train, y_train)

# Making predictions
predictions = best_model.predict(X_test)

# Evaluating the model (mean squared error)
mse = sum((true - pred)**2 for true, pred in zip(y_test, predictions)) / len(y_test)
print("Mean Squared Error:", mse)

# Function to write data to CSV file
def write_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

# Function to read data from CSV file
def read_from_csv(filename):
    with open(filename, 'r') as csvfile:
        data = [line.strip().split(',') for line in csvfile.readlines()]
    return data

# Function to collect sensor data
def collect_sensor_data():
    # Simulate sensor data collection
    sensor_data = []
    for _ in range(100):
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        temperature = random.uniform(20, 40)
        humidity = random.uniform(30, 70)
        solar_irradiance = random.uniform(200, 1000)
        battery_voltage = random.uniform(48, 54)
        sensor_data.append([date, temperature, humidity, solar_irradiance, battery_voltage])
    return sensor_data

# Function to analyze sensor data
def analyze_sensor_data(sensor_data):
    # Calculate average temperature and humidity
    total_temperature = 0
    total_humidity = 0
    for row in sensor_data:
        total_temperature += float(row[1])
        total_humidity += float(row[2])
    average_temperature = total_temperature / len(sensor_data)
    average_humidity = total_humidity / len(sensor_data)
    print("Average Temperature:", average_temperature)
    print("Average Humidity:", average_humidity)

# Function to perform model training and evaluation
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Train linear regression model
    slope, intercept = linear_regression_fit(X_train, y_train)
    
    # Make predictions
    predictions = linear_regression_predict(X_test, slope, intercept)
    
    # Evaluate model
    mse = sum((true - pred)**2 for true, pred in zip(y_test, predictions)) / len(y_test)
    print("Mean Squared Error:", mse)

# Main function
def main():
    # Collect sensor data
    sensor_data = collect_sensor_data()
    
    # Write sensor data to CSV file
    write_to_csv(sensor_data, 'sensor_data.csv')
    
    # Read sensor data from CSV file
    sensor_data_from_csv = read_from_csv('sensor_data.csv')
    
    # Analyze sensor data
    analyze_sensor_data(sensor_data_from_csv)
    
    # Extract features and target variable from sensor data
    X = [[float(row[1]), float(row[2]), float(row[3]), float(row[4])] for row in sensor_data_from_csv[1:]]
    y = [random.uniform(0, 1) for _ in range(len(sensor_data_from_csv) - 1)]  # Dummy target variable
    
    # Split data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train and evaluate model
    train_and_evaluate_model(X_train, X_test, y_train, y_test)


main()

# Modify collect_sensor_data() function to collect data from additional sensors or sources
def collect_sensor_data():
    # Add code to collect data from additional sensors or sources
    pass

# Data preprocessing (handling missing values, outliers, normalization, etc.) can be done here
# Example: Replace missing values with mean
def preprocess_data(sensor_data):
    # Add preprocessing steps here
    return sensor_data

# Example feature engineering: Creating new features
def feature_engineering(sensor_data):
    # Add code to create new features
    return sensor_data

# Example model selection: Decision Tree Regression
class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):

        n_samples = len(X)
        variance = self._variance(y)
        best_variance_reduction = 0
        best_split = None

        for feature_index in range(len(X[0])):
            values = [X[i][feature_index] for i in range(n_samples) if isinstance(X[i][feature_index], (int, float))]
            if not values:
                continue  # Skip feature if all values are non-numeric

            thresholds = sorted(set(values))
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2.0
                left_indices = [j for j in range(n_samples) if X[j][feature_index] <= threshold]
                right_indices = [j for j in range(n_samples) if X[j][feature_index] > threshold]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_variance = self._variance([y[j] for j in left_indices])
                right_variance = self._variance([y[j] for j in right_indices])
                weighted_variance = (len(left_indices) / n_samples) * left_variance + (len(right_indices) / n_samples) * right_variance
                variance_reduction = variance - weighted_variance

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }

        if best_variance_reduction > 0 and (self.max_depth is None or depth < self.max_depth):
            left = self._build_tree([X[i] for i in best_split['left_indices']], [y[i] for i in best_split['left_indices']], depth + 1)
            right = self._build_tree([X[i] for i in best_split['right_indices']], [y[i] for i in best_split['right_indices']], depth + 1)
            return Node(feature_index=best_split['feature_index'], threshold=best_split['threshold'], left=left, right=right)
        else:
            return Node(value=sum(y) / len(y))

    def _variance(self, arr):
        mean = sum(arr) / len(arr)
        return sum((x - mean) ** 2 for x in arr) / len(arr)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

# Instantiate model
model = DecisionTreeRegressor(max_depth=3)

# Example hyperparameter tuning: Manual tuning
# You can manually set hyperparameters when instantiating the model
model = DecisionTreeRegressor(max_depth=3)

# Function to calculate Mean Squared Error (MSE)
def mean_squared_error(actual, predicted):
    n = len(actual)
    squared_errors = [(actual[i] - predicted[i]) ** 2 for i in range(n)]
    mse = sum(squared_errors) / n
    return mse

# Actual values (ground truth)
actual_values = [10, 20, 30, 40, 50]

# Predicted values from your model
predicted_values = [12, 18, 32, 35, 48]

# Calculate Mean Squared Error
mse = mean_squared_error(actual_values, predicted_values)

# Interpretation of MSE
print("Mean Squared Error:", mse)
if mse == 0:
    print("Perfect prediction: The predicted values match the actual values exactly.")
elif mse < 10:
    print("Low error: The model's predictions are very close to the actual values.")
elif 10 <= mse < 100:
    print("Moderate error: The model's predictions are somewhat close to the actual values, but there is room for improvement.")
else:
    print("High error: The model's predictions are significantly different from the actual values. Further investigation and improvement are needed.")
