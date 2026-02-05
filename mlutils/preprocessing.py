import io
from PIL import Image
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List
import matplotlib.pyplot as plt

def run_kalman_filter(measurements, process_noise_covariance, measurement_noise_covariance, dt):
  """
  Runs a Kalman Filter on a list of 1D measurements.

  Args:
    measurements (list or np.array): A list or array of 1D scalar measurements.
    process_noise_covariance (np.array): The process noise covariance matrix (Q).
    measurement_noise_covariance (np.array): The measurement noise covariance matrix (R).
    dt (float): The time step between measurements.

  Returns:
    np.array: An array of estimated states (positions) after filtering.
  """
  if not isinstance(measurements, np.ndarray):
    measurements = np.array(measurements)

  if measurements.ndim > 1 and measurements.shape[1] > 1:
    raise ValueError("This Kalman Filter is configured for 1D scalar measurements only.")

  # Initialize Kalman Filter for a 1D constant velocity model
  # State vector x: [position, velocity]
  # Measurement z: [position]

  kf = KalmanFilter(dim_x=2, dim_z=1)

  # State transition matrix (F) for a constant velocity model
  # position_k = position_{k-1} + velocity_{k-1} * dt
  # velocity_k = velocity_{k-1}
  kf.F = np.array([[1., dt[0]],
                   [0., 1.]])

  # Measurement function (H) - maps state space to measurement space
  # We only observe the position
  kf.H = np.array([[1., 0.]])

  # Process noise covariance (Q)
  # This represents uncertainty in the state model itself.
  # Small values for a relatively smooth process.
  kf.Q = np.array([[process_noise_covariance,0],
                   [0,process_noise_covariance]])

  # Measurement noise covariance (R)
  # This represents uncertainty in the sensor measurements.
  kf.R = np.array([[measurement_noise_covariance]]) # Scalar for 1D measurement noise

  # Initial state estimate (x)
  # Assume initial position is the first measurement, initial velocity is 0
  kf.x = np.array([[measurements[0]],
                   [0.]])

  # Initial state covariance (P)
  # Large values to represent high initial uncertainty
  kf.P = np.array([[1000., 0.],
                   [0., 1000.]])

  # Lists to store the filtered states
  filtered_states = []
  velocity_states = []
  for i in range(len(measurements)):
    kf.F = np.array([[1., dt[i]],
                     [0., 1.]])
    kf.predict()
    kf.update(measurements[i])
    # Append scalar values from kf.x to avoid object dtype issues in subsequent processing
    filtered_states.append(kf.x[0, 0]) # Extract the scalar position
    velocity_states.append(kf.x[1, 0]) # Extract the scalar velocity

  return np.array(filtered_states), np.array(velocity_states)

def min_max_scaling(data):
    # Use numpy's min and max functions which work on both pandas objects and numpy arrays
    min_val = np.min(data)
    max_val = np.max(data)

    min_val = min_val if min_val != 0 else 0.0000001

    # Handle the case where max_val - min_val is zero to prevent division by zero
    denominator = (max_val - min_val)
    if denominator == 0:
        # If all values are the same, return an array of zeros.
        return np.zeros_like(data, dtype=float)

    return (data - min_val) / denominator

def generate_features(df):
    """
    Generates Kalman filtered features and normalizes them.

    Args:
    df (pd.DataFrame): The input DataFrame containing 'Close' prices and 'UtcTimestamp'.


    Returns:
    pd.DataFrame: A DataFrame of normalized features.
    """
    # Generate kalman filters and stack for input
    prices = df["Close"].to_list()

    # Adjusted variances to better demonstrate filtering
    variances = ( # (process_noise, measurement_noise)
        (0.001, 1000.0),   # Very smooth
        (0.1, 1000.0),   # Moderately smooth
        (10.0, 1000.0),   # Less smooth
        (10.0, 10.0),   # Follows measurements closely
        (0.00001, 1000.0), # Extremely smooth
        (0.0000001, 1000.0)  # Captures General movement
    )
    input_velocities = [[0.00000000001 for i in range(len(prices))]]
    input_filtered_prices = [prices]
    for v in variances:
        filtered_prices, velocities = run_kalman_filter(prices, v[0], v[1], df['dt'])
        input_filtered_prices.append(filtered_prices)
        input_velocities.append(velocities)
    if len(filtered_prices) != len(velocities):
        raise ValueError("Mismatch length 0f filtered_prices is not equal to length of velocities")
    # chart
    start = 300
    end = 500
    x = [i for i in range(len(prices))]
    plt.figure(figsize=(12, 7))
    for i in range(len(variances)+1):
        if i == 0:
            plt.plot(x[start:end], input_filtered_prices[0][start:end], label='Original Prices', linewidth=2)
            continue
        plt.plot(x[start:end], input_filtered_prices[i][start:end], label=f'KF (Q={variances[i-1][0]}, R={variances[i-1][1]}) - Position', linestyle='--')
    plt.title('Min-Max Scaled Kalman Filtered Positions for Different Variances')
    plt.xlabel('Time Step')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    filtered_prices = pd.DataFrame(input_filtered_prices).T
    velocities = pd.DataFrame(input_velocities).T
    return (filtered_prices, velocities)

def generate_images(df, sequence_length, display_images):
    """Function for generating a sequence of images from Kalman Filter and prices data"""
    print("Geerating features...")
    features = generate_features(df)
    filtered_prices = features[0]
    velocities = features[1]
    # display("Filtered Prices:", filtered_prices.head())
    # display("Velocities:", velocities.head())
    # Gather sequneces, converts (,no_of_features) to (,sequence_length,no_of_features)
    # The reshape method returns a new array. It needs to be assigned back to features.
    # Using -1 for the first dimension lets numpy infer the correct size.
    sequenced_features = []
    for i in range(len(velocities) - sequence_length):
        # applying min max scaling seperately because they contain different values
        price_feature = min_max_scaling(filtered_prices.iloc[i:i+sequence_length])
        vel_feature = min_max_scaling(velocities.iloc[i:i+sequence_length])
        # if i == 10:
        #     display("Price Feature:", price_feature)
        #     display("Vel Feature:", vel_feature)
        sequenced_features.append(np.concatenate((price_feature, vel_feature), axis=1))
    features = np.array(sequenced_features)
    print("Sequenced Features Shape:", features.shape)
    images = []
    for i in range(len(features)):
        image = np.array(features[i])
        # image = min_max_scaling(features[i] @ features[i].T)
        images.append(image)
        if i in display_images:
            # display(image)
            plt.imshow(image)
            if display_images.index(i) == 0:
                title = "BUY"
            elif display_images.index(i) == 1:
                title = "NEUTRAL"
            else:
                title = "SELL"
            plt.title(title)
            plt.axis('off') # Hide axes ticks and labels
            plt.show()
    images = np.array(images)
    print("Images Shape:", images.shape)
    return images.reshape(-1, 1, sequence_length, 14)

def label_dataset(prices, horizon):
    """Takes in DataFrame and generates dataset based on criteria, returns a list of [0,0,1]"""
    labels = []
    count = [0,0,0]
    examples = [0,0,0]
    for i in range(len(prices) - horizon):
        if np.all(prices.iloc[i:i+horizon] > prices.iloc[i] - 1) and np.any(prices.iloc[i:i+horizon] > prices.iloc[i] + 2):
            labels.append([0,0,1])
            count[2] += 1
            examples[2] = i
        elif np.all(prices.iloc[i:i+horizon] < prices.iloc[i] + 1) and np.any(prices.iloc[i:i+horizon] < prices.iloc[i] - 2):
            labels.append([1,0,0])
            count[0] += 1
            examples[0] = i
        else:
            labels.append([0,1,0])
            count[1] += 1
            examples[1] = i
    print("Counts up: {} down: {} neutral: {}".format(count[2],count[0],count[1]))
    return np.array(labels), examples

def read_dataframe_from_json(file_path):
  """
  Reads a DataFrame from a JSON file.

  Args:
    file_path (str): The path to the JSON file.

  Returns:
    pd.DataFrame: The DataFrame read from the JSON file.
  """
  try:
    df = pd.read_json(file_path)
    return df
  except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
    return None
  except Exception as e:
    print(f"An error occurred while reading the JSON file: {e}")
    return None

def generate_dt(t):
    print("UtcTimestamps:")
    display(t.head())
    dt = [0]
    x0 = None
    for x in t:
        if x0 is None:
            x0 = x
            continue
        dt.append((x-x0)/1000) # convert to seconds by dividing by 1000
        x0 = x
    if len(dt) != len(t):
        raise ValueError("Mismatch length 0f t is not equal to length of x0")
    return np.array(dt)

from torch.utils.data import IterableDataset

# Generates input data for training and validation
class InputGenerator(IterableDataset):
    def __init__(self, filenames, sequence_length, horizon):
        self.filenames = filenames
        self.sequence_length = sequence_length
        self.horizon = horizon

    def __iter__(self):
        for filename in self.filenames:
            # Your existing code
            df = pd.DataFrame(read_dataframe_from_json('/content/datasets/'+filename)["data"].to_list())
            df['dt'] = generate_dt(df['UtcTimestamp'])
            print("Generating images...")
            labels, examples = label_dataset(df["Close"].shift(-self.sequence_length).dropna(), self.horizon)
            images = generate_images(df, sequence_length=self.sequence_length, display_images=examples)[:-self.horizon]
            # Convert to tensors and yield INDIVIDUAL samples
            for i in range(len(images)):
                yield (
                    torch.FloatTensor(images[i]),
                    torch.FloatTensor(labels[i])
                )

    def save(self, file_path):
        all_images = []
        all_labels = []
        for filename in self.filenames:
            df = pd.DataFrame(read_dataframe_from_json('/content/datasets/'+filename)["data"].to_list())
            df['dt'] = generate_dt(df['UtcTimestamp'])
            print(f"Generating images for {filename}...")
            labels, examples = label_dataset(df["Close"].shift(-self.sequence_length).dropna(), self.horizon)
            images = generate_images(df, sequence_length=self.sequence_length, display_images=examples)[:-self.horizon]

            print(images.shape, labels.shape)
            # Extend the lists with generated data
            all_images.extend([torch.FloatTensor(img) for img in images])
            all_labels.extend([torch.FloatTensor(lbl) for lbl in labels])

        # Stack all images and labels into single tensors
        stacked_images = torch.stack(all_images)
        stacked_labels = torch.stack(all_labels)

        # Save to file
        torch.save({'images': stacked_images, 'labels': stacked_labels}, file_path)
        print(f"Dataset saved to {file_path}")

def train_one_epoch(epoch_index, training_loader):
    cutmix = v2.CutMix(num_classes=3)
    mixup = v2.MixUp(num_classes=3)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = cutmix_or_mixup(inputs, labels)
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('\r  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss
