from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Beispiel: Zeitreihen-Daten
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Hyperparameter-Suchraum
l2_values = [0.01, 0.025, 0.05, 0.08]
learning_rates = [0.001, 0.002, 0.004, 0.006, 0.008]
patience_values = [72]
batch_sizes = [32]

# Ergebnis speichern
best_params = None
best_loss = float("inf")

# Schleife über den Suchraum
for l2_value in l2_values:
    for lr in learning_rates:
        for patience in patience_values:
            for batch_size in batch_sizes:
                print(f"Testing with L2: {l2_value}, LR: {lr}, Patience: {patience}, Batch size: {batch_size}")
                
                # Cross-Validation
                fold_losses = []
                for train_index, val_index in tscv.split(X):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                    # Modell definieren
                    model = Sequential([
                        Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(l2_value)),
                        Dropout(0.15),
                        Dense(256, activation='relu'),
                        Dropout(0.38),
                        Dense(128, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(1, activation='relu'),
                    ])
                    
                    # Optimizer und Kompilierung
                    optimizer = Adam(learning_rate=lr)
                    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

                    # Early Stopping Callback
                    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

                    # Trainieren
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=800,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    # Validierungsergebnis
                    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
                    fold_losses.append(val_loss)

                # Durchschnittlicher Loss für alle Folds
                mean_loss = np.mean(fold_losses)
                print(f"Mean Validation Loss: {mean_loss}")

                # Beste Hyperparameter speichern
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_params = {
                        'l2': l2_value,
                        'learning_rate': lr,
                        'patience': patience,
                        'batch_size': batch_size
                    }

# Ergebnisse
print("Best Hyperparameters:")
print(best_params)
print(f"Best Loss: {best_loss}")

# Trainieren mit den besten Hyperparametern auf Train+Validation und Test auf Testdaten
final_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(best_params['l2'])),
    Dropout(0.15),
    Dense(256, activation='relu'),
    Dropout(0.38),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='relu'),
])

final_optimizer = Adam(learning_rate=best_params['learning_rate'])
final_model.compile(optimizer=final_optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
final_early_stopping = EarlyStopping(patience=best_params['patience'], restore_best_weights=True)

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=800,
    batch_size=best_params['batch_size'],
    callbacks=[final_early_stopping],
    verbose=1
)

# Testdaten auswerten
test_loss = final_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss[0]}, Test MAE: {test_loss[1]}, Test MSE: {test_loss[2]}")


######################################################################

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Beispiel: Zeitreihen-Daten
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Hyperparameter-Suchraum
l2_values = [0.01, 0.025, 0.05, 0.08]
learning_rates = [0.001, 0.002, 0.004, 0.006, 0.008]
patience_values = [72]
batch_sizes = [32]

# Ergebnis speichern
best_params = None
best_loss = float("inf")

# Schleife über den Suchraum
for l2_value in l2_values:
    for lr in learning_rates:
        for patience in patience_values:
            for batch_size in batch_sizes:
                print(f"Testing with L2: {l2_value}, LR: {lr}, Patience: {patience}, Batch size: {batch_size}")
                
                # Cross-Validation
                fold_losses = []
                for train_index, val_index in tscv.split(X):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                    # Modell definieren
                    model = Sequential([
                        Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(l2_value)),
                        Dropout(0.15),
                        Dense(256, activation='relu'),
                        Dropout(0.38),
                        Dense(128, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(1, activation='relu'),
                    ])
                    
                    # Optimizer und Kompilierung
                    optimizer = Adam(learning_rate=lr)
                    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

                    # Early Stopping Callback
                    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

                    # Trainieren
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=800,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    # Validierungsergebnis
                    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
                    fold_losses.append(val_loss)

                # Durchschnittlicher Loss für alle Folds
                mean_loss = np.mean(fold_losses)
                print(f"Mean Validation Loss: {mean_loss}")

                # Beste Hyperparameter speichern
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_params = {
                        'l2': l2_value,
                        'learning_rate': lr,
                        'patience': patience,
                        'batch_size': batch_size
                    }

# Ergebnisse
print("Best Hyperparameters:")
print(best_params)
print(f"Best Loss: {best_loss}")

# Trainieren mit den besten Hyperparametern auf Train+Validation und Test auf Testdaten
final_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(best_params['l2'])),
    Dropout(0.15),
    Dense(256, activation='relu'),
    Dropout(0.38),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='relu'),
])

final_optimizer = Adam(learning_rate=best_params['learning_rate'])
final_model.compile(optimizer=final_optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
final_early_stopping = EarlyStopping(patience=best_params['patience'], restore_best_weights=True)

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=800,
    batch_size=best_params['batch_size'],
    callbacks=[final_early_stopping],
    verbose=1
)

# Testdaten auswerten
test_loss = final_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss[0]}, Test MAE: {test_loss[1]}, Test MSE: {test_loss[2]}")


######################################################################





from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Beispiel: Zeitreihen-Daten
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Hyperparameter-Suchraum
l2_values = [0.01, 0.025, 0.05, 0.08]
learning_rates = [0.001, 0.002, 0.004, 0.006, 0.008]
patience_values = [72]
batch_sizes = [32, 16]

# Ergebnis speichern
best_params = None
best_loss = float("inf")

# Schleife über den Suchraum
for l2_value in l2_values:
    for lr in learning_rates:
        for patience in patience_values:
            for batch_size in batch_sizes:
                print(f"Testing with L2: {l2_value}, LR: {lr}, Patience: {patience}, Batch size: {batch_size}")
                
                # Cross-Validation
                fold_losses = []
                for train_index, val_index in tscv.split(X):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                    # Modell definieren

                    model = Sequential([
                    Dense(256, activation='relu', input_shape=(Data.shape[1],), kernel_regularizer=l2(l2_value)),
                    Dropout(0.25),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(1, activation='linear'),
                    ])
                    # Optimizer und Kompilierung
                    optimizer = Adam(learning_rate=lr)
                    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

                    # Early Stopping Callback
                    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

                    # Trainieren
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=800,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=0
                    )

                    # Validierungsergebnis
                    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
                    fold_losses.append(val_loss)

                # Durchschnittlicher Loss für alle Folds
                mean_loss = np.mean(fold_losses)
                print(f"Mean Validation Loss: {mean_loss}")

                # Beste Hyperparameter speichern
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_params = {
                        'l2': l2_value,
                        'learning_rate': lr,
                        'patience': patience,
                        'batch_size': batch_size
                    }

# Ergebnisse
print("Best Hyperparameters:")
print(best_params)
print(f"Best Loss: {best_loss}")

# Trainieren mit den besten Hyperparametern auf Train+Validation und Test auf Testdaten
final_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(best_params['l2'])),
    Dropout(0.15),
    Dense(256, activation='relu'),
    Dropout(0.38),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='relu'),
])

final_optimizer = Adam(learning_rate=best_params['learning_rate'])
final_model.compile(optimizer=final_optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
final_early_stopping = EarlyStopping(patience=best_params['patience'], restore_best_weights=True)

final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=800,
    batch_size=best_params['batch_size'],
    callbacks=[final_early_stopping],
    verbose=1
)

# Testdaten auswerten
test_loss = final_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss[0]}, Test MAE: {test_loss[1]}, Test MSE: {test_loss[2]}")

# stopped at an hour running time, nice in theory, but we don't have the computing power to run this efficiently
