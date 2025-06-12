import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/admin/Downloads/Bank Customer Churn Prediction.csv')

# Preprocessing
# Drop customer_id as it's not a useful feature
data = data.drop('customer_id', axis=1)

# Separate features and target
X = data.drop('churn', axis=1)
y = data['churn']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define categorical and numerical features
categorical_features = ['country', 'gender', 'credit_card', 'active_member']
numerical_features = ['credit_score', 'age', 'tenure', 'balance', 
                     'products_number', 'estimated_salary']

# Create preprocessing pipelines for DNN and Wide & Deep
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Fit and transform the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get the number of features after preprocessing
num_features = X_train_processed.shape[1]

# Function to create a basic DNN model
def create_dnn_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model

# Function to create a Wide & Deep model
def create_wide_deep_model(input_dim):
    # Wide part (linear model)
    input_layer = layers.Input(shape=(input_dim,))
    wide = layers.Dense(1, activation='linear')(input_layer)
    
    # Deep part
    deep = layers.Dense(128, activation='relu')(input_layer)
    deep = layers.Dropout(0.3)(deep)
    deep = layers.Dense(64, activation='relu')(deep)
    deep = layers.Dropout(0.2)(deep)
    deep = layers.Dense(32, activation='relu')(deep)
    
    # Combine wide and deep
    combined = layers.concatenate([wide, deep])
    output = layers.Dense(1, activation='sigmoid')(combined)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model

# Function to create a TabTransformer model
def create_tabtransformer_model():
    # Numerical features
    num_inputs = layers.Input(shape=(len(numerical_features),), name='num_input')
    num_x = layers.Dense(32, activation='relu')(num_inputs)
    num_x = layers.Dense(32, activation='relu')(num_x)
    
    # Categorical features
    cat_inputs = []
    cat_embeddings = []
    for cat_col in categorical_features:
        unique_vals = X_train[cat_col].nunique()
        input_layer = layers.Input(shape=(1,), name=f'{cat_col}_input')
        embedding = layers.Embedding(
            input_dim=unique_vals + 1, 
            output_dim=32)(input_layer)
        embedding = layers.Reshape(target_shape=(32,))(embedding)
        cat_inputs.append(input_layer)
        cat_embeddings.append(embedding)
    
    # Stack categorical embeddings
    cat_x = layers.Concatenate(axis=1)(cat_embeddings)
    cat_x = layers.Reshape((len(categorical_features), 32))(cat_x)  # Shape: (batch, num_cat_features, embedding_dim)
    
    # Transformer part for categorical features
    cat_x = layers.LayerNormalization()(cat_x)
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=32)(cat_x, cat_x)
    cat_x = layers.Add()([cat_x, attention_output])
    cat_x = layers.LayerNormalization()(cat_x)
    cat_x = layers.Flatten()(cat_x)
    cat_x = layers.Dense(32, activation='relu')(cat_x)
    
    # Combine numerical and categorical features
    combined = layers.Concatenate(axis=1)([num_x, cat_x])
    
    # MLP
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(
        inputs=[num_inputs] + cat_inputs,
        outputs=output
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model

# Prepare data for TabTransformer
def prepare_tabtransformer_data(X):
    # Split into numerical and categorical
    num_data = X[numerical_features].values
    cat_data = [X[col].values.reshape(-1, 1) for col in categorical_features]
    
    # Scale numerical data
    scaler = StandardScaler()
    num_data = scaler.fit_transform(num_data)
    
    # Encode categorical data (simple integer encoding)
    cat_encoded = []
    for i, col in enumerate(categorical_features):
        unique_vals = X_train[col].unique()
        val_to_idx = {v: i+1 for i, v in enumerate(unique_vals)}
        encoded = np.array([val_to_idx[v] for v in X[col]]).reshape(-1, 1)
        cat_encoded.append(encoded)
    
    return [num_data] + cat_encoded

# Prepare TabTransformer data
X_train_tt = prepare_tabtransformer_data(X_train)
X_test_tt = prepare_tabtransformer_data(X_test)

# Train and evaluate models
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    early_stopping = keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class))
    
    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred))
    
    return history

# Model 1: DNN
print("Training DNN Model...")
dnn_model = create_dnn_model(num_features)
dnn_history = train_and_evaluate(dnn_model, X_train_processed, y_train, 
                                X_test_processed, y_test)

# Model 2: Wide & Deep
print("\nTraining Wide & Deep Model...")
wide_deep_model = create_wide_deep_model(num_features)
wide_deep_history = train_and_evaluate(wide_deep_model, X_train_processed, y_train, 
                                      X_test_processed, y_test)

# Model 3: TabTransformer
print("\nTraining TabTransformer Model...")
tabtransformer_model = create_tabtransformer_model()
tabtransformer_history = train_and_evaluate(tabtransformer_model, X_train_tt, y_train, 
                                           X_test_tt, y_test)

# Plot training history
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('AUC')
    plt.legend()
    
    plt.show()

plot_history(dnn_history, "DNN Model")
plot_history(wide_deep_history, "Wide & Deep Model")
plot_history(tabtransformer_history, "TabTransformer Model")