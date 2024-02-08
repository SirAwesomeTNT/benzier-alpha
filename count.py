# count.py

# Define a function to count from 1 to 10
def count_to_10():
    for i in range(1, 11):
        # Send each number to the calling script
        yield i
