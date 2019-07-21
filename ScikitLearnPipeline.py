# Import our data =============================================================
import sklearn
data = sklearn.datasets.load_boston(return_X_y=False)

# Get X, y data sets
import pandas as pd
# Get X
X = pd.DataFrame(data = data.data,
                  columns = data.feature_names)
# Get y
y = pd.DataFrame(data = data.target,
                 columns = ["target"])

# Bring together for full datasaet
data = pd.concat([X,y], axis = 1)
# Examine data ================================================================
data.head(10)

# Missing Values ==============================================================
data.isnull().sum()

# Plot ========================================================================
import seaborn as sns
sns.set(style = "ticks")

data.plot.scatter("CRIM", "target")

# Fit Simple Linear Regression Model ==========================================
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Train test split data
simple_x = data["CRIM"]
X_train, X_test, y_train, y_test = train_test_split(simple_x, y, test_size = 0.3, random_state = 27)

# Initialize linear regression model
reg = LinearRegression()

# Fit model to training data
reg.fit(X_train[:,None], y_train)

# Predict test set
pred = reg.predict(X_test[:,None])

# Score predictions
reg.score(X_test[:,None],y_test)

# With log transform ==========================================================
import numpy as np

# Log transform the CRIM column
logtransform_x = np.log(data["CRIM"])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(logtransform_x, y, test_size = 0.3, random_state = 27)

# Initialize model
lt_reg = LinearRegression()

# Fit Model
lt_reg.fit(X_train[:,None], y_train)

# Predict test data
pred = lt_reg.predict(X_test[:,None])

# Score predictions
lt_reg.score(X_test[:,None], y_test)

# A slight improvement




# Build the transform into a custom pipeline ==================================
from sklearn.base import TransformerMixin

# Define custom log transform
class LogTransformP1(TransformerMixin):
    def __init__(self):
        self.lt = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xlog = np.log(X + 0.01)
        return Xlog


# Create Pipeline =============================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# Numerical Features ==========================================================
numFeats = ["CRIM"]

# Pipeline
numericTransformer = Pipeline(steps = [
        ("LogTransform", LogTransformP1()),
        ("scaler", StandardScaler())])

# Categorical Features ========================================================
catFeats = ["CHAS"]

categoricalTransformer = Pipeline(steps = [
        ("OneHot", OneHotEncoder(handle_unknown='ignore'))])

# Column Transformer ==========================================================
preprocessor = ColumnTransformer(
        transformers = [
                ('num', numericTransformer, numFeats),
                ('cat', categoricalTransformer, catFeats)])

# Pipeline ====================================================================
pipeline = Pipeline(steps = [
        ("preprocessor", preprocessor),
        ("classifier", LinearRegression())])

# Grid Search For Parameters ==================================================
# params = {
#        'preprocessor__num__scaler__parameters': ["example1", "example2"]}
    
# gscv = GridSearchCV(pipeline, params, cv=5)

# Train Test Split ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 27)

# Fit Model ===================================================================

pipeline.fit(X_train, y_train)

print("Model Score: {}%".format(round(100* pipeline.score(X_test,y_test),2)))




    