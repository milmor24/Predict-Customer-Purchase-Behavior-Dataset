int(input("Choose a option: ").strip())

while True:
    print("Choose a option\n(1) Load Dataset\n(2) Train a Classification Model\n(3) Evaluate and Save the Performance\n(4) Simulate enviroment")
    option = int(input("Choose a option: ").strip())

    if option == 1:
          df = load_dataset
