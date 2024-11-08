import numpy as np #type: ignore
import torch# type: ignore
import torch.nn as nn# type: ignore
import torch.optim as optim# type: ignore
from torch.utils.data import DataLoader, TensorDataset# type: ignore
import pandas as pd# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.model_selection import KFold # type: ignore
import matplotlib.pyplot as plt# type: ignore

def load_data(csv_file, target_columns):
    """Load dataset from CSV and apply consistent outlier removal with a fallback."""
    df = pd.read_csv(csv_file)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle and reset index
    X = df.drop(columns=target_columns).values
    y = df[target_columns].values
    return X, y

def filter_test_set(X_train, y_train, X_test, y_test):
    """Remove any rows in test set that overlap with training data."""
    train_df = pd.DataFrame(X_train)
    train_df['target'] = pd.DataFrame(y_train).apply(tuple, axis=1) 

    test_df = pd.DataFrame(X_test)
    test_df['target'] = pd.DataFrame(y_test).apply(tuple, axis=1)

    # Remove test records that match any training record
    filtered_test_df = test_df[~test_df.isin(train_df)].dropna()
    
    X_test_filtered = filtered_test_df.drop(columns='target').values
    y_test_filtered = np.array([list(t) for t in filtered_test_df['target']])
    
    return X_test_filtered, y_test_filtered

#Modello
#Inizializzazione dei pesi
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)  

#modello effettivo
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Allenamento con valutazione su test set ad ogni epoca
class EarlyStopping:
    def __init__(self, patience, min_delta=0, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.epochs_no_improve = 0
            self.save_checkpoint(model)
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Save the model when the validation loss improves."""
        torch.save(model.state_dict(), self.save_path)
        print("Model saved with improved validation loss.")

def train_model(model, train_loader, criterion, optimizer, X_val, y_val, epochs):
    early_stopping = EarlyStopping(patience=10, save_path="best_model.pth")

    loss_history= [0]*epochs

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation at the end of each epoch
        model.eval()
        with torch.no_grad():
            val_predictions = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_predictions, torch.FloatTensor(y_val))

        loss_history[epoch]= round(val_loss.item(), 4)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

        # Early stopping check
        early_stopping(val_loss.item(), model)
        
        # Stop if early stopping criterion is met
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            for i in range(epoch+1, epochs):
                loss_history[i] = loss_history[epoch]
            return loss_history

    # Load the best model before returning
    model.load_state_dict(torch.load("best_model.pth", weights_only=False))
    return loss_history


#Valutazione modello
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(X_test))
        y_test_tensor = torch.FloatTensor(y_test)
        # Mean Squared Error (MSE)
        mse_loss_fn = nn.MSELoss()
        mse = mse_loss_fn(predictions, y_test_tensor)
        print(f'Mean Squared Error: {mse.item():.4f}')

        # Mean Absolute Error (MAE)
        mae_loss_fn = nn.L1Loss()
        mae = mae_loss_fn(predictions, y_test_tensor)
        print(f'Mean Absolute Error: {mae.item():.4f}')
        
        # Coefficient of Determination (R^2)
        y_mean = torch.mean(y_test_tensor)
        ss_total = torch.sum((y_test_tensor - y_mean) ** 2)
        ss_residual = torch.sum((y_test_tensor - predictions) ** 2)
        r2 = 1 - ss_residual / ss_total
        print(f'R squared Score: {r2.item():.4f}')

        # Si ritornano tutte le metreiche per confronti finali
        return {
            "MSE": mse.item(),
            "MAE": mae.item(),
            "R2": r2.item()
        }
    

def main(train_csv_file, test_csv_file, target_columns, epochs, n_splits):
    X_train_full, y_train_full = load_data(train_csv_file, target_columns)
    X_test_full, y_test_full = load_data(test_csv_file, target_columns)
    print(f"length of train file {len(X_train_full)}")
    print(f"length of test file {len(X_test_full)}")

    X_train_scaling, _, y_train_scaling, _ = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=2)
    X_combined_scaling = np.vstack((X_train_scaling, X_test_full))
    scaler = MinMaxScaler()
    X_combined_scaling = scaler.fit_transform(X_combined_scaling)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    fold = 1
    cv_metrics = {"MSE": [], "MAE": [], "R2": []}
    test_metrics = {"MSE": [], "MAE": [], "R2": []}

    #per il calcolo dell'andamento della loss
    all_historyes = []

    for train_index, val_index in kf.split(X_train_scaling):
        #print(f"\n\n--- Fold {fold} ---")
        fold += 1
        X_train, X_val = X_train_scaling[train_index], X_train_scaling[val_index]
        y_train, y_val = y_train_scaling[train_index], y_train_scaling[val_index]
        X_test, y_test = filter_test_set(X_train_scaling, X_train_scaling, X_test_full, y_test_full)
        print(f"length of train file {len(X_train)}")
        print(f"length of test file {len(X_test)}")

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        input_size = X_train.shape[1]
        output_size = y_train.shape[1]
        model = FeedForwardNN(input_size, output_size)
        model.apply(initialize_weights)

        criterion = nn.L1Loss()  # MAE Loss
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        loss_history = train_model(model, train_loader, criterion, optimizer, X_val, y_val, epochs)
        all_historyes.append(loss_history)

        print("andamento loss:")
        print(loss_history)

        #print(f"\nresults of fold on validation")
        fold_metrics = evaluate_model(model, X_val, y_val)
        cv_metrics["MSE"].append(fold_metrics["MSE"])
        cv_metrics["MAE"].append(fold_metrics["MAE"])
        cv_metrics["R2"].append(fold_metrics["R2"])

        #print(f"\nresults of fold on test")
        test_fold_metrics = evaluate_model(model, X_test, y_test)
        test_metrics["MSE"].append(test_fold_metrics["MSE"])
        test_metrics["MAE"].append(test_fold_metrics["MAE"])
        test_metrics["R2"].append(test_fold_metrics["R2"])

    print(f"\n\nFinal results training with {train_csv_file}")
    print("\n\n--- Cross-Validation Results on Validation Set ---")
    print(f"Average MSE: {sum(cv_metrics['MSE']) / n_splits:.4f}")
    print(f"Average MAE: {sum(cv_metrics['MAE']) / n_splits:.4f}")
    print(f"Average R2 Score: {sum(cv_metrics['R2']) / n_splits:.4f}")

    print("\n--- Evaluation on Separate Test Set ---")
    print(f"Average MSE: {sum(test_metrics['MSE']) / n_splits:.4f}")
    print(f"Average MAE: {sum(test_metrics['MAE']) / n_splits:.4f}")
    print(f"Average R2 Score: {sum(test_metrics['R2']) / n_splits:.4f}")

    all_historyes = np.array(all_historyes)
    average_progression = np.mean(all_historyes, axis=0)

    # Plot the average loss progression
    # plt.plot(average_progression, label="loss media", color="blue")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    # plt.ylim(ymin=4, ymax = 14)
    # plt.xlim(xmin=0, xmax = 40)
    # plt.title("Andamento medio della loss")
    # plt.legend()
    # plt.show()

test_csv_file = 'FinalDatasetNoNan.csv'
for i in range(14):
    train_csv_file = ['LongRec2.csv','LongRec2boOut.csv',
                      'LowEgs.csv','ZsOutLowEgs.csv', 'PrOutLowEgs.csv', 'boOutLowEgs.csv',
                      'MidEGs.csv','ZsOutMidEGs.csv', 'PrOutMidEGs.csv', 'boOutMidEGs.csv',
                      'HigEGs.csv','ZsOutHigEGs.csv', 'PrOutHigEGs.csv', 'boOutHigEGs.csv'][i]
    test_csv_file  = ['LongRec2.csv','LongRec2.csv',
                      'LowEgs.csv','LowEgs.csv', 'LowEgs.csv', 'LowEgs.csv',
                      'MidEGs.csv','MidEGs.csv', 'MidEGs.csv', 'MidEGs.csv',
                      'HigEGs.csv','HigEGs.csv', 'HigEGs.csv', 'HigEGs.csv'][i]
    target_columns = ['VolumeNP', 'Glucosio50', 'Aminoacidi06', 'Lipidi20']
    print(f"Target columns: {target_columns}")
    print(f"\n\nTraining with {train_csv_file} and testing with {test_csv_file}")
    main(train_csv_file, test_csv_file, target_columns, epochs=40, n_splits=4)


