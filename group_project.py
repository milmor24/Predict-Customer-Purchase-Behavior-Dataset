import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, classification_report # For evaluating model performance

class CPP: # Customer Purchase Prediction class, includes all methods
     def __init__(self):
          self.df = None 
          self.model = None
          self.predictions = None
          self.features_test = None
          self.features_train = None
          self.purchase_status_train = None
          self.purchase_status_test = None

     def load_dataset(self): 
          ds_name = input("Type the full name of the dataset: ").strip() 
          try: 
               self.df = pd.read_csv(ds_name) 
               print(f"Loaded CSV file: {ds_name}") 
               print(f"The first 10 rows: \n{self.df.head(10)}\n") 
               print(f"Basic statistics: \n{self.df.describe()}\n") 
          except FileNotFoundError: # If no such file exists
               print("File not found. Check the path and try again.")

     def train_classification_module(self): 
          if self.df is None: # If no dataset is loaded
               print("No dataset loaded.")
               return 

          self.purchase_status = self.df["PurchaseStatus"] # Making "Purchase_Status" the target variable
          self.features = self.df.drop("PurchaseStatus", axis=1) # Features without the target variable
          # Splitting the dataset into training and testing sets to be used later
          self.features_train, self.features_test, self.purchase_status_train, self.purchase_status_test = train_test_split(
          self.features, self.purchase_status, test_size=0.2, random_state=50, stratify=self.purchase_status
          ) # 80-20 split with stratification(to maintain even class distribution)

          module = input("Choose a module: \n(1) K-NN\n(2) Decision Tree\n").strip() 
          if module == "1": 
               k_value = int(input("Enter the value of k for K-NN: ")) # Getting k value from user
               self.model = KNeighborsClassifier(n_neighbors=k_value) # Running K-NN with user-defined k
               print("Training K-NN model...")
          elif module == "2": 
               self.model = DecisionTreeClassifier(random_state=50) 
               print("Training Decision Tree model...")
          else:
               print("Invalid module.")
               return

          self.model.fit(self.features_train, self.purchase_status_train) # Fitting the features and target variable to the model
          self.predictions = self.model.predict(self.features_test) # Making predictions on the test set
          print("Model training complete!") 

     def evaluate_save_performance(self): 
        if self.model is None: 
          print("No trained model found, please train a model first!")
          return 
        choice = input("Do you want to evaluate using a new dataset file? (yes/no): ").strip().lower() 
        if choice == "yes": 
            new_ds = input("Enter the new dataset filename: ").strip() 
            try: 
                 new_df = pd.read_csv(new_ds)
                 if "PurchaseStatus" not in new_df.columns: 
                      print("The new dataset must include the 'PurchaseStatus' column.")
                      return
                 new_features = new_df.drop("PurchaseStatus", axis=1) # Features without the target variable for new dataset
                 new_purchase_status = new_df["PurchaseStatus"] # Target variable for new dataset
                 # Splitting the new dataset into training and testing sets to be used later
                 new_features_train, new_features_test, new_purchase_status_train, new_purchase_status_test = train_test_split(
                 new_features, new_purchase_status, test_size=0.2, random_state=50, stratify=new_purchase_status
                 ) # 80-20 split with stratification(to maintain even class distribution)
                 self.predictions = self.model.predict(new_features_test) # Making predictions on new dataset's test features

            except FileNotFoundError: 
                 print("No such file found.")
                 return 
        else: 
             if self.predictions is None or self.purchase_status_test is None: 
                  print("No previously trained model or predictions found.")
                  return 
             new_purchase_status = self.purchase_status_test 
             self.predictions = self.model.predict(self.features_test) 
        # Show results
        acc = accuracy_score(new_purchase_status, self.predictions) # Calculating accuracy as 'acc'
        report = classification_report(new_purchase_status, self.predictions) # Generating classification report as 'report'
        print(f"\nAccuracy: {acc:.4f}\n") 
        print("Classification Report:\n", report) 

        # Asking user if they want to save the report
        save_choice = input("Do you want to save the performance report to a .txt file? (yes/no): ").strip().lower()
        if save_choice == "yes": 
             filename = input("Enter the filename to save the report (with .txt extension): ").strip()
             with open(filename, "w") as file: 
                  file.write("Performance Report\n")
                  file.write("------------------\n")
                  file.write(f"Accuracy: {acc:.4f}\n\n")
                  file.write("Classification Report:\n")
                  file.write(report)
             print(f"Performance report saved to {filename}") 
        else: # If user does not want to save the report
             print("Performance report not saved.")
                          

     def simulate_environment(self): 
          if self.model is None: 
               print("You need to train a model first.")
               return
          if self.df is None: 
               print("No dataset loaded.")
               return

          print("\nSimulate Environment")
          # Prompting user for new customer data
          print("Enter numeric values for a new customer:\n" \
          "Gender: Male = 0, Female = 1, loyalty program: No = 0, Yes = 1\n" \
          "Categories: Electronics = 0, Clothing = 1, Home Goods = 2, Beauty = 3, Sports = 4\n")

          # Getting all feature names except the target column into a list called 'feature_cols'
          feature_cols = [col for col in self.df.columns if col != "PurchaseStatus"]
          new_data = [] # List to hold new customer data

          # Collect input values for each feature
          for col in feature_cols: # Iterating through each feature column
               new_value = input(f"{col}: ").strip() # Getting user input for each feature and stripping whitespace
               try:
                    new_value = float(new_value) # Converting input to float
               except ValueError: # If conversion fails, we notify the user and exit
                    print(f"Invalid input for {col}. Please enter a numeric value.")
                    return
               new_data.append(new_value) 

          input_df = pd.DataFrame([new_data], columns=feature_cols) 

          prediction = self.model.predict(input_df)[0] # Making prediction for the new customer data
          print(f"\nPredicted Purchase Status: {int(prediction)} (1 = Purchase, 0 = No Purchase)")


      
print(
    "Welcome to this program for Predicting Customer Purchase Behavior.\n"
    "By evaluating customer data such as Gender, AnnualIncome (in dollars), NumberOfPurchases, ProductCategory,\n"
    "TimeSpentOnWebsite (in minutes), LoyaltyProgram, and DiscountsAvailed, we can predict whether a customer\n"
    "is likely to make a purchase or not.\n"
    "Choose an option from the menu below to get started!"
)

cpp = CPP()
while True:
    
     print("Menu:\n(1) Load Dataset\n(2) Train a Classification Model\n(3) Evaluate and Save the Performance\n"
     "(4) Simulate environment\n(9) Quit and lose data:")
     try:
          choice = int(input("Enter the number of your choice: "))
     except ValueError:
          print("Please enter a valid number.\n")
          continue

     match choice:
          case 1:
               cpp.load_dataset()
          case 2:
               cpp.train_classification_module()
          case 3:
               cpp.evaluate_save_performance()
          case 4:
               cpp.simulate_environment()
          case 9:
               print("You have exited the program")
               break
          case _:
               print("Invalid option, try again")

