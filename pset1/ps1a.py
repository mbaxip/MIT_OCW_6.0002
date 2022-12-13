###########################
# 6.0002 Problem Set 1a: Space Cows 
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
import time
import numpy

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    cows = {}
    file = open(filename, "r")
    cow = str(file.readline())
    while cow:
        name = cow.split(",")[0]
        weight = cow.split(",")[1].rstrip()
        cows[name] = weight
        cow = file.readline()
    file.close()
    return cows

# Problem 2
def greedy_cow_transport(cows,limit):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """

    # Sort the cows in the descending order of their weights.
    cow_names = list(cows.keys())
    cow_weights = list(cows.values())
    sort_id = numpy.argsort(cow_weights)[::-1]
    sorted_cow_weights = []
    sorted_cow_names = []
    for i in range(len(sort_id)):
        sorted_cow_weights.append(int(cow_weights[sort_id[i]]))
        sorted_cow_names.append(cow_names[sort_id[i]])

    trips = []
    while sorted_cow_names:
        trip = []
        new_limit = 10
        # For each trip pick the heaviest cow first until the limit is exhausted
        i = 0
        while i < len(sorted_cow_names):
            # Check if the weight of the next heaviest cow is within the weight limit allowed for the trip
            if sorted_cow_weights[i] <= new_limit:
                # Add the cow to the trip
                trip.append(sorted_cow_names[i])
                # Update the remaining weight limit of the trip
                new_limit = new_limit - sorted_cow_weights[i]
                # Remove the cow added to the trip from the list
                sorted_cow_names.pop(i)
                sorted_cow_weights.pop(i)
            else:
                # If the next heaviest cow does not fit, move to the next cow
                i = i + 1
        trips.append(trip)
    return trips


# Problem 3
def brute_force_cow_transport(cows,limit):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    j = 0
    res = []
    # Loop over all possible combinations of cows and number of trips
    # First checks all cows in one trip, then combinations of cows in two trips and so on....
    for partition in get_partitions(cows.keys()):

        # Finds the first combination of cows and trips (partition) that is not over the weight limit,
        # minimizing the number of trips.
        trip_flag = 1
        # Loop over all the trips in a combination (partition)
        for trip in partition:
            trip_weight = 0
            # Loop over all the cows in a trip
            for cow in trip:
                trip_weight = trip_weight + int(cows[cow])
            # Check if the total weight of cows in a trip is over the weight limit
            if trip_weight > limit:
                trip_flag = 0
                break
        # Check if the combination of trips satisfies the criteria of cows in each trip being under the weight limit
        if trip_flag == 1 and j == 0:
            res = partition
            j = 1
        elif len(res) > len(partition) and trip_flag == 1:
            res = partition
    # Found the right combination of cows and trips.
    return res


        
# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    # Load data
    cows = load_cows("ps1_cow_data.txt")
    # weight limit for each trip
    limit = 10

    # Run Greedy Algorithm
    start = time.time()
    trips_greedy = greedy_cow_transport(cows, limit)
    end = time.time()
    greedy_time = end - start
    print("Greedy Algo Time (in secs): " + str(greedy_time))
    print("Greedy trips: " + str(trips_greedy))

    # Run Brute Force Algorithm
    start = time.time()
    trips_brute = brute_force_cow_transport(cows, limit)
    end = time.time()
    greedy_time = end - start
    print("Brute Force Algo Time (in secs): " + str(greedy_time))
    print("Brute Force trips: " + str(trips_brute))


compare_cow_transport_algorithms()
