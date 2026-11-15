import pandas as pd
import torch

x = torch.randn(100, 10)
df = pd.DataFrame(x.numpy())
df.to_csv('data.csv', index=False, float_format='%.2f')
