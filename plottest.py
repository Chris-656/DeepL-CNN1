import matplotlib.pyplot as plt

# Days of the week
days = [1, 2, 3, 4, 5, 6, 7]

# Number of apples eaten each day
apples_eaten = [1, 2, 1, 3, 2, 2, 1]

# Create a line chart
plt.plot(days, apples_eaten)

# Add a title
plt.title('Apples Eaten Per Day')

# Label the x-axis
plt.xlabel('Day')

# Label the y-axis
plt.ylabel('Apples Eaten')

# Show the plot
plt.show()