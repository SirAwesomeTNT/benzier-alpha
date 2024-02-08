# run.py
import runpy

# Execute the count.py script
count_script_globals = runpy.run_path("count.py")
print(count_script_globals)

# Access the function from count.py
count_function = count_script_globals['count_to_10']

# Call the function and print each number
for number in count_function():
    print(number)
