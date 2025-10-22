import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
df = None
print("---MENU FOR DATA ANALYSTICS---")

def load_dataset():
     ds_name = input("Type the name of the dataset (CSV): ").strip()
     try:
          df = pd.read_csv(ds_name)
          print(f"Loaded CSV file: {ds_name}")
          print(f"This is the DH-Head \n{df.head()}")
          return df
     except FileNotFoundError:
          print("File not found. Check the path and try again.")


def train_classification_module(df):
     print("Choose a module: \n(1) KNN\n(2)Decision Tree")
     module = input("Choose a module: ").strip()
     if module == "1":
          # Train KNN model
          knn = KNeighborsClassifier()
          knn.fit(X_train, y_train)
          print("Trained KNN model.")
     elif module == "2":
          # Train Decision Tree model
          dt = DecisionTreeClassifier()
          dt.fit(X_train, y_train)
          print("Trained Decision Tree model.")
     else:
          print("Invalid module.")

def evaluate_save_perfomance(df):

def simulate_enviroment(df):
     None

      

while True:
    
     print("Choose an option\n(1) Load Dataset\n(2) Train a Classification Model\n(3) Evaluate and Save the Performance\n(4) Simulate enviroment (5) Q for quit:")
     option = input("Choose an option: ").strip().lower()

     if option == "1":
          df = load_dataset()
     elif option == "2":
          if df is None:
               print("You have to load dataset first!")
          else:
               try:
                    train_classification_module(df)
               except: print("An error has accured")


     elif option == "q":
          print("You have exited the program")
          break
     else:
          print("Invalid option, try again")

