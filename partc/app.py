from regression import regression_models
from classification import Classification_model
import pandas as pd


import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

def main():

    print('----------------------------------------------------------------------')    
    regression_models()
    print("\n")
    print('-----------------------------------------------------------------------')
    Classification_model()
    print("\n")


if __name__ == "__main__":
    main()