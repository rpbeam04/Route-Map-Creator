#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import random
from pprint import *

# Set the URL to scrape
url = "https://en.wikipedia.org/wiki/List_of_airports_in_the_United_States"

# Define a filename for the HTML file to be saved
filename = "Wikipedia_Airports"

# Check if the file has been saved already today
if os.path.exists(filename):
    print("I have a file!")
    # If the file exists, open it and read its contents into the html variable
    with open(filename, "r") as f:
        html = f.read()
else:
    # If the file doesn't exist, scrape the URL and save the HTML file
    response = requests.get(url)
    print("I'm scraping!")
    html = response.text
    with open(filename, "w") as f:
        f.write(html)

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(html, "html.parser")

# Find all the tables in the HTML
tables = soup.find_all("table")

# Load each table into a pandas data frame
dfs = []
for table in tables:
    # Convert the table HTML to a pandas dataframe and append it to the list of dataframes
    df = pd.read_html(str(table))[0]
    dfs.append(df)
print(f"{len(dfs)} tables found.")
if len(dfs) == 1:
    # If only one table was found, use that as the dataframe
    df = dfs[0]


# In[2]:


# Get the third table from the list of dataframes
airports = dfs[2]

# Add a new column called "State" and fill it with NaN values
airports["State"] = np.nan

# Loop through each row in the airports dataframe
for index,row in airports.iterrows():
    # Check if the "Airport" column for the current row is NaN
    airports_nan = airports.isna()
    if airports_nan.loc[index,"Airport"] == True:
        # If it is NaN, set the "state" variable to the value in the "City" column for the current row
        state = airports.loc[index,"City"]
    else:
        # If it is not NaN, set the "State" column for the current row to the "state" variable
        airports.loc[index,"State"] = str(state.title())

# Drop any rows where the "Airport" column is NaN
airports.dropna(subset=["Airport"],inplace=True)

# Print the first five rows of the dataframe
airports.head()


# In[3]:


# Read in the "us-airports.csv" file as a pandas dataframe
us_airports = pd.read_csv('us-airports.csv')

# Merge the "airports" and "us_airports" dataframes using the "ICAO" column in "airports" and the "ident" column in "us_airports"
all_airports = pd.merge(airports, us_airports, left_on='ICAO', right_on='ident')

# Drop the "ident" column from the merged dataframe
all_airports.drop('ident', axis=1, inplace=True)

# Delete the "airports" and "us_airports" dataframes to free up memory
del(airports)
del(us_airports)

# Capitalize the first letter of each column name in the merged dataframe
all_airports.columns = all_airports.columns.str.title()

# Rename the "Faa", "Iata", and "Icao" columns to "FAA", "IATA", and "ICAO", respectively
all_airports.rename(columns={"Faa":"FAA","Iata":"IATA","Icao":"ICAO"},inplace=True)

# Print the first five rows of the merged dataframe
all_airports.head()


# In[4]:


class Airport:
    def __init__(self, row):
        for col in row.index:
            setattr(self, col, row[col])
            
    def __repr__(self):
        return f"Airport({self.ICAO})" # Returns a string representation of the Airport object
        
    def __str__(self):
        return f"{self.ICAO}" # Returns a string representation of the Airport object
    
    Airport_List = [] # A list to store Airport objects


# In[5]:


for _, row in all_airports.iterrows():
    Airport.Airport_List.append(Airport(row)) # Create Airport objects and add them to Airport_List
    
pprint(Airport.Airport_List) # Pretty-print the list of Airport objects


# In[6]:


class Route:
    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination
    
    def __repr__(self):
        return f"Route({self.origin.ICAO} -> {self.destination.ICAO})"
        
    def __str__(self):
        return f"{self.origin.ICAO} -> {self.destination.ICAO}"

def create_random_route_network(num_routes, hubs=None):
    if hubs is None:
        hubs = []

    # Convert the enplanements to a logistic distribution for weighting
    weights = np.logistic(0, 100000, size=len(Airport.Airport_List))

    # Get a list of possible origin airports
    origin_list = [airport for airport in Airport.Airport_List if airport.ICAO not in hubs]

    routes = []

    while len(routes) < num_routes:
        # Select a random origin airport based on weights
        origin = random.choices(origin_list, weights=weights)[0]

        # Get a list of possible destination airports
        dest_list = [airport for airport in Airport.Airport_List if airport.ICAO not in hubs and airport != origin]

        # Select a random destination airport based on weights
        destination = random.choices(dest_list, weights=weights)[0]

        # Check if the origin and destination airports have at least one other route
        if (len(origin.routes) > 1 or len(dest.routes) > 1) and origin != destination:
            # Create the route
            route = Route(origin, destination)
            routes.append(route)
            origin.routes.append(route)
            dest.routes.append(route)

    # Check for isolated routes and repeat until there are none
    while True:
        isolated_routes = []
        for airport in Airport.Airport_List:
            if len(airport.routes) == 1:
                isolated_routes.append(airport.routes[0])

        if len(isolated_routes) == 0:
            break

        for route in isolated_routes:
            origin = route.origin
            dest = route.destination
            origin.routes.remove(route)
            dest.routes.remove(route)
            routes.remove(route)

        while len(routes) < num_routes:
            origin = random.choices(origin_list, weights=weights)[0]
            dest_list = [airport for airport in Airport.Airport_List if airport.ICAO not in hubs and airport != origin]
            destination = random.choices(dest_list, weights=weights)[0]
            if (len(origin.routes) > 1 or len(dest.routes) > 1) and origin != destination:
                route = Route(origin, destination)
                routes.append(route)
                origin.routes.append(route)
                dest.routes.append(route)

    return routes


# In[7]:


from scipy.special import expit

# Get enplanements for all airports
enplanements = [airport.Enplanements for airport in Airport.Airport_List]

# Normalize enplanements to a logistic curve
enplanements_logistic = expit((enplanements - np.mean(enplanements)) / np.std(enplanements))

# Assign Weight attribute to each airport based on enplanements
for i, airport in enumerate(Airport.Airport_List):
    airport.Weight = enplanements_logistic[i]
    
# create empty dictionary to store weights
weights_dict = {}

# iterate over the list of airports
for airport in Airport.Airport_List:
    # add airport and its weight to the dictionary
    weights_dict[airport.ICAO] = airport.Weight


# In[8]:


class Route:
    def __init__(self, origin, destination):
        self.origin = origin
        self.destination = destination
        self.isolated = False

    def __repr__(self):
        return f"Route({self.origin}-{self.destination})"


# In[9]:


def search_ICAO(icao_code):
    for airport in Airport.Airport_List:
        if airport.ICAO == icao_code:
            return airport
    return None


# In[10]:


def create_random_route_network(num_routes, hubs=None, hub_modifier=4, selection_modifier=0.5):
    """
    Creates a random route network for an airline.

    Parameters:
    num_routes (int): Number of routes to create.
    hubs (list of str, optional): List of ICAO code strings representing hubs.
    
    Returns:
    list of Route: List of randomly generated routes.
    """
    if hubs is None:
        hubs = []
        
    weights = weights_dict.copy()
    
    for hub in hubs:
        weights[hub] = weights[hub]*hub_modifier
    
    airport_list = [airport.ICAO for airport in Airport.Airport_List]
    route_list = []
    
    while len(route_list) < num_routes:
        weights_list = list(weights.values())
        origin = random.choices(airport_list, weights_list)[0]
        destination = origin
        while destination == origin:
            destination = random.choices(airport_list, weights_list)[0]

        route_list.append(Route(origin, destination))
        weights[origin] = weights[origin]+selection_modifier
        weights[destination] = weights[destination]+selection_modifier
        
    origins = [route.origin for route in route_list]
    destinations = [route.destination for route in route_list]
    cities = origins+destinations
    counts = dict(Counter(cities))
    isolated = []
    for city, count in counts.items():
        if count == 1:
            isolated.append(city)
            
    for route in route_list:
        if (route.origin in isolated) and (route.destination in isolated):
            route_list.remove(route)

    print(f"Generated {len(route_list)} routes.")
    return route_list


# In[11]:


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_route_map(routes):
    # Create a basemap of the US
    fig = plt.figure(figsize=(10, 8))
    m = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=50, llcrnrlon=-130, urcrnrlon=-60, resolution='h')
    m.drawcoastlines()
    m.drawcountries()
    
    # Plot airports as dots
    for route in routes:
        start_airport = route.Start
        end_airport = route.End
        start_lat = float(start_airport.Latitude_Deg)
        start_lon = float(start_airport.Longitude_Deg)
        end_lat = float(end_airport.Latitude_Deg)
        end_lon = float(end_airport.Longitude_Deg)
        x1, y1 = m(start_lon, start_lat)
        x2, y2 = m(end_lon, end_lat)
        m.plot([x1, x2], [y1, y2], 'ro-', markersize=5, linewidth=1)
        
    plt.show()


# In[12]:


from collections import Counter

routes = create_random_route_network(100, hubs = ['KCLT','KLGA'])
origins = [route.origin for route in routes]
destinations = [route.destination for route in routes]
cities = origins+destinations

counts = Counter(cities)
pprint(counts)

for route in routes:
    print(route.origin,route.destination)


# In[13]:


plot_route_map(routes)


# In[ ]:




