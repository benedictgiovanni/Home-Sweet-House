import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Contoh data pelatihan
data = pd.read_csv('jadifix.csv')

# Cetak nama kolom untuk verifikasi
print(data.columns)

# Misalnya data memiliki fitur berikut
numeric_features = ['MSSubClass', 'LotArea', 'YearBuilt', 'Id']
categorical_features = ['SaleCondition']

# Preprocessing untuk data numerik dan kategorik
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Buat pipeline yang menyertakan preprocessing dan model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Pisahkan fitur dan label
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Latih model
model.fit(X, y)

# Simpan pipeline
joblib.dump(model, 'model/pipeline.pkl')
