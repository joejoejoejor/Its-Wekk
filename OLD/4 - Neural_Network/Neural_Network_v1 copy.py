# Import necessary packages
#import cmocean # Nice colormaps
import numpy as np # Numerical computation package
import pandas as pd # Dataframe package
#import matplotlib.pyplot as plt # Plotting package
np.random.seed(1) # Set the random seed for reproduceability
from sys import stdout


# Read in the dataset
DF = pd.read_csv(r"C:\Users\maxd2\OneDrive - Universitaet St.Gallen\Dokumente\GitHub\Its-Wekk\5 - Models\Neural_Network\Data\Working_DataFrame_NN.csv")
# Drop useless columns
DF.drop(columns=["Weekday"], inplace=True)



X = np.array(DF).astype(np.float16)

TargetDF = pd.read_csv(r"C:\Users\maxd2\OneDrive - Universitaet St.Gallen\Dokumente\GitHub\Its-Wekk\5 - Models\Neural_Network\Data\Final_Target_Data_Combined_resid_Trend.csv")

TargetDF = TargetDF[TargetDF['Datum'] <= '2024-10-20 21:00:00+00:00']

# no need to drop the date column
#TargetDF = TargetDF.drop(columns=["Datum"], inplace=True)

# Get one-hot-encoded target
Y = np.array(pd.get_dummies(TargetDF["PM10_Combined_Trend_Residual"])).astype(np.float16)

Y = TargetDF["PM10_Combined_Trend_Residual"].values.reshape(-1, 1).astype(np.float16)


# Initialize some parameters
N, d = X.shape # Number of observations and features
L = Y.shape[1] # Number of outputs

sl = 16 # Number of hidden nodes
epochs = 100 # Number of training epochs
eta = 0.01 # Learning rate
init_range = [-.5, .5] # The range of our uniform distribution for weight initialization



# Initialize weight matrices
np.random.seed(72) # Set seed
W1 = np.random.rand(sl, d) # Randomly initialize W1
W1 = W1 * (init_range[1] - init_range[0]) - init_range[0] # Constrain to range
#W2 = np.random.rand(L, sl) # Randomly initialize W2
#W2 = W2 * (init_range[1] - init_range[0]) - init_range[0] # Constrain to range
W2 = np.random.uniform(init_range[0], init_range[1], size=(1, sl))  # Shape (1, sl)


# We will use the sigmoid function as the activation function
sigmoid = lambda x: 1 / (1 + np.exp(-x))


def forward_pass(X, W1, W2):
    try:
        print("Step 1: Start forward pass")

        # Compute the matrix multiplication of the input layer
        S = X @ W1.T
        print("Step 2: After input layer multiplication, S shape:", S.shape)
 
        # Pass through the non-linear activation function
        Z = sigmoid(S)
        print("Step 3: After activation function, Z shape:", Z.shape)

        # Compute the matrix multiplication of the hidden layer
        T = Z @ W2.T
        print("Step 4: After hidden layer multiplication, T shape:", T.shape)

        # Pass through the non-linear activation function 
        # (This should always be sigmoid for the output to be (0, 1))
        # ⚠️ Notice how we don't flatten the output anymore, this time we want to keep it a matrix!
        output = sigmoid(T)
        print("Step 5: Final output computed, output shape:", output.shape)

        return output
    except Exception as e:
            print("Error occurred:", str(e))
            raise


# Check that the output of a forward pass is indeed an NxL matrix
pd.DataFrame(forward_pass(X, W1, W2), columns=[f"Y{i}" for i in range(L)])


############################################################################################################################################################################

def eval_predictions(X, Y, W1, W2):
    # Use the forward_pass function defined above to compute our estimated probability
    prob = forward_pass(X, W1, W2)
    
    # To avoid log(0), clip the probability to not be exactly zero or one
    prob = np.clip(prob, 1e-8, 1-1e-8)
    # Calculate the loss function (negative log-likelihood in this case)
    #loss = - np.sum(Y * np.log(prob) + (1 - Y) * np.log(1 - prob))
    #use the MSE for Loss
    loss = np.mean((Y - prob) ** 2)


    # For the actual prediction, we select the outcome with the highest probability. 
    # Unlike in the previous case, the two probabilities need not add up to 1!
    pred = prob.argmax(1) # Take the argmax along the second axis
    
    # Compute number of misclassification
    misclassifications = np.mean(pred != Y.argmax(1))
    
    # Output results as a dictionary
    return {
        "loss": loss, 
        "misclassifications": misclassifications, 
        "prob": prob, 
        "pred": pred
    }

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)




# Initializitation of lists for bookkeeping
loss_list = []
misclassification_list = []

# Create an array of indices (which we will shuffle later on) through which we 
# will iterate. This represent the index of the observations
indices = np.arange(N)





np.random.seed(72) # Reset random seed for reproduceability
# Compute the loss and misclassifications BEFORE training
res = eval_predictions(X, Y, W1, W2)

# Append to our result lists
loss_list.append(res["loss"])
misclassification_list.append(res["misclassifications"])

total_mse = 0

# Run the full training loop (iterate over the number of training epochs)
for epoch in range(epochs):
    # Reshuffle the indices
    np.random.shuffle(indices)
    
    # Iterate through each single data point
    for i in indices:
        # Note that we use [i:i+1, :] instead of [i, :] to keep it as a 1x2 matrix
        Xi = X[i:i+1, :] # Extract features for ith observation,
        Yi = Y[i:i+1, :] # Extract label for ith observation
        
        # ----- Forward pass -----
        # Computes the predictions using Xi, W1, W2. Here we avoid using the
        # forward_pass function because we need the hidden nodes values Zi
        # for backpropagation. So, instead, we repeat the code (ugh...)
        
        # Pass to the hidden nodes (pre-activation)
        Si = Xi @ W1.T
        # Compute activation function
        Zi = sigmoid(Si)
        
        # Pass to the output nodes
        Ti = Zi @ W2.T
        # Compute sigmoid probability transformation
        prob_i = sigmoid(Ti)
        # ⚠️ Notice that we now have two probabilities!!
        

        #MSE
        mse_i = mean_squared_error(Yi, prob_i)
        total_mse += mse_i
        
        # ----- Backward pass -----
        # ⚠️ Since we have two output nodes, and two probabilities for prediction
        # we will also have two errors!!
        error_i = Yi - prob_i
        
        # Compute the gradient w.r.t. W2. ⚠️ W2 is a vector! See math derivations
        grad2_i = -Zi.T @ error_i # ⚠️ error_i is now 1 x L, and Zi is sl x 1
        # Compute the gradient w.r.t. W1. ⚠️ W1 is matrix! Making things even worse
        grad1_i = -Xi.T @ ((error_i @ W2) * Zi * (1 - Zi)) # ⚠️ error_i is now 1 x L
        
        # Updating: Move 'eta' units in the direction of the negative gradient
        W1 -= (eta * grad1_i).T
        W2 -= (eta * grad2_i).T
    
    # Evaluate the learning process and store the results into our lists
    res = eval_predictions(X, Y, W1, W2)
    loss_list.append(res["loss"])
    misclassification_list.append(res["misclassifications"])    
        
    # Print the current status ( ignore this part!)
    bar = "".join(["#" if epoch >= t * (epochs // 50) else " " for t in range(50)])
    stdout.write(f"\rEpoch: {epoch+1:>{int(np.floor(np.log10(epochs))+1)}}/{epochs} [{bar}]")
    avg_mse = total_mse / len(indices)
    print(f"Epoch {epoch+1}/{epochs}, MSE: {avg_mse:.4f}")






    # Set up the canvas
#fig, axs = plt.subplots(1, 2, figsize=(16, 8))
# Plot the loss over the epochs (epoch 0 is before training!)
#axs[0].plot(range(len(loss_list)), loss_list)
# Plot the misclassification rate over the epochs
#axs[1].plot(range(len(misclassification_list)), misclassification_list)
# Add title, grid, axis labels
#for ax in axs:
#    ax.grid(True)
#    ax.set_xlabel("Epoch number")
#axs[0].set_ylabel("Loss (negative log-likelihood)")
#axs[0].set_title("Evolution of loss function over training epochs")
#axs[1].set_ylabel("Misclassification rate")
#axs[1].set_title("Evolution of missclassification rate over training epochs")


print(W1.T.shape)

# Display the weight matrix from the input layer to the hidden layer
pd.DataFrame(W1.T, columns=[f"Z{i}" for i in range(sl)], index=[f"X{i+1}" for i in range(d)])




# Display the weight matrix  from the hidden layer to the output layer
pd.DataFrame(W2.T, columns=[f"Y{i}" for i in range(L)], index=[f"Z{i}" for i in range(sl)])