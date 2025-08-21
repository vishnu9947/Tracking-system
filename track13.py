import threading

def print_numbers():
    for i in range(5):
        print(i)

# Creating a thread
thread = threading.Thread(target=print_numbers)

# Starting the thread
thread.start()

# Waiting for the thread to complete
thread.join()

print("Thread execution finished.")
