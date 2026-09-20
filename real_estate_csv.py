import csv
import random as rd

filename = "real_estate.csv"
features = ["area_in_sqft", "price"]
realestate = []

for i in range(1, 1001):
    data = {
        "area_in_sqft": rd.randint(2000, 5000),
        "price": rd.randint(5000000, 7500000)
    }
    realestate.append([data["area_in_sqft"], data["price"]])

# Write to CSV
with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow( features)
    writer.writerows(realestate)
