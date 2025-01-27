import numpy as np
import pandas as pd

# Generate linear data
X = np.linspace(0,10,100).reshape(-1,1)
y = 3 * X.squeeze() + np.random.randn(100) * 0.5

# Save as CSV
data = pd.DataFrame({"X": X.squeeze(), "y": y})
data.to_csv("data/synthetic_data.csv", index=False)