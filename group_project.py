import pandas as pd
df = None
print("---MENU FOR DATA ANALYSTICS---")

def load_dataset():
     ds_name = input("Type the name of the dataset (CSV): ").strip()
     try:
          df = pd.read_csv(ds_name)
     except:


def train_classification_moduke(df):
      
def evaluate_save_perfomance(df):

def simulate_enviroment(df):
     None

      

while True:
    print("Choose a option\n(1) Load Dataset\n(2) Train a Classification Model\n(3) Evaluate and Save the Performance\n(4) Simulate enviroment")
    option = int(input("Choose a option: ").strip())

    if option == 1:
          df = load_dataset()



