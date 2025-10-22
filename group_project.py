from unittest import case
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

class CPP: # Customer Purchase Prediction
     def __init__(self):
          self.df = None # DataFrame to hold the dataset
          self.model = None # Machine learning model
          self.predictions = None # Model predictions
          self.purchase_status_test = None # True labels for the test set
          self.features_test = None # Features for the test set
          self.features_train = None # Features for the training set
          self.purchase_status_train = None # True labels for the training set
          self.purchase_status_test = None # True labels for the test set

     def load_dataset(self):
          ds_name = input("Type the full name of the dataset: ").strip()
          try:
               self.df = pd.read_csv(ds_name)
               print(f"Loaded CSV file: {ds_name}")
               print(f"The first 10 rows: \n{self.df.head(10)}")
          except FileNotFoundError:
               print("File not found. Check the path and try again.")

     def train_classification_module(self):
          if self.df is None:
               print("No dataset loaded.")
               return # Exit if no dataset is loaded

          purchase_status = self.df["PurchaseStatus"] # Making "Purchase_Status" the target variable
          features = self.df.drop("PurchaseStatus", axis=1) # Features without the target variable
          features_train, features_test, purchase_status_train, purchase_status_test = train_test_split(
          features, purchase_status, test_size=0.2, random_state=50, stratify=purchase_status
          ) # Splitting the dataset into training and testing sets to be used later
          
          module = input("Choose a module: \n(1) K-NN\n(2) Decision Tree\n").strip()
          if module == "1": # Train K-NN model
               k_value = int(input("Enter the value of k for K-NN: "))
               self.model = KNeighborsClassifier(n_neighbors=k_value)
               print("Training K-NN model...")
          elif module == "2": # Train Decision Tree model
               self.model = DecisionTreeClassifier(random_state=50)
               print("Training Decision Tree model...")
          else:
               print("Invalid module.")

          # Train and predict
          self.model.fit(features_train, purchase_status_train)
          self.predictions = self.model.predict(features_test)

          # Save for evaluation
          self.purchase_status_test = purchase_status_test
          self.features_test = features_test
          self.features_train = features_train
          self.purchase_status_train = purchase_status_train

          print("Model training complete!")

     def evaluate_save_performance(self):
        if self.predictions is None or self.purchase_status_test is None:
          print("No trained model or predictions found.")
          return

        filename = input("Enter the .txt filename to save the performance report: ").strip()
        with open(filename, "w") as file:
          file.write("Performance Report\n")
          file.write("------------------\n")
          file.write(f"Accuracy: {accuracy_score(self.purchase_status_test, self.predictions):.4f}\n\n")
          file.write("Classification Report:\n")
          file.write(classification_report(self.purchase_status_test, self.predictions))

          print(f"Performance report saved to {filename}")

     def simulate_environment(df):
          None

      
print("---Predicting Customer Purchase Behavior---")

cpp = CPP()
while True:
    
     print("Choose an option\n(1) Load Dataset\n(2) Train a Classification Model\n"
     "(3) Evaluate and Save the Performance\n(4) Simulate environment\n(9) Quit and lose data:")
     choice = int(input("Enter the number of your choice: "))

     match choice:
          case 1:
               cpp.load_dataset()
               df = cpp.df
          case 2:
               cpp.train_classification_module()
          case 3:
               cpp.evaluate_save_performance()
          case 4:
               simulate_environment(df)
          case 9:
               print("You have exited the program")
               break
          case _:
               print("Invalid option, try again")

